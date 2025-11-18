# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
from collections.abc import Sequence
from enum import Enum
import time

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pydantic import BaseModel
from elasticsearch import AsyncElasticsearch

from nat.data_models.intermediate_step import IntermediateStepType
from nat.builder.framework_enum import LLMFrameworkEnum

from aiva_agent.observability.nemo_dfw.converters import otel_span_to_dfw_record
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import  NemoDFWRecord

logger = logging.getLogger(__name__)


class ExportStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SCHEDULED = "scheduled"


class NemoDFWElasticsearchExporter(SpanExporter):
    """AgentIQ trace exporter to NeMo Data Flywheel Intake service."""

    def __init__(self,
                 elasticsearch_url="http://localhost:9200",
                 elasticsearch_auth=("elastic", "changeme"),
                 client_id="aiq-traces",
                 index="aiq-traces"):
        self._elastic_client = AsyncElasticsearch(
            elasticsearch_url,
            basic_auth=elasticsearch_auth,
            headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"})
        self._client_id = client_id
        self._index = index
        self._loop = asyncio.get_running_loop()

    async def _export_span(
        self, span: ReadableSpan
    ) -> None:
        
        try:
            match span.attributes.get("aiq.event_type"):
                case IntermediateStepType.LLM_START:
                    # Skip non-LangChain LLM events until support for other frameworks is added
                    if span.attributes["aiq.framework"] not in (LLMFrameworkEnum.LANGCHAIN.value):
                        logger.warning("Skipping non-LangChain LLM event.")
                        return
                    converted_payload = otel_span_to_dfw_record(span)
                    if converted_payload is None:
                        return
                    # Set client_id
                    converted_payload.client_id = self._client_id
                    await self._elastic_client.index(index=self._index, body=converted_payload.model_dump(by_alias=True))
                case _:
                    pass
            
        except Exception as e:
            logger.exception("Error exporting NeMo data flywheel telemetry traces: %s", e, exc_info=e)

    def export(
        self, spans: Sequence[ReadableSpan]
    ) -> SpanExportResult:
        """Exports a batch of telemetry data.

        Args:
            spans: The list of `opentelemetry.trace.Span` objects to be exported

        Returns:
            SpanExportResult: The result of the export, always returns SpanExportResult.SUCCESS.
        """

        try:
            for span in spans:
                self._loop.create_task(self._export_span(span=span))
                logger.debug("NeMo data flywheel export status: %s", ExportStatus.SCHEDULED.name)
            
            return SpanExportResult.SUCCESS
        
        except Exception as e:
            logger.exception("Error exporting NeMo data flywheel telemetry traces: %s", e, exc_info=e)
        
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shuts down the exporter.

        Called when the SDK is shut down.
        """

        logger.info("Shutting down NeMo Data Flywheel trace exporter...")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Nothing is buffered in this exporter, so this method does nothing."""

        return True
