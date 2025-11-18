# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import FunctionRef

logger = logging.getLogger(__name__)


class StructuredRagFnConfig(FunctionBaseConfig, name="structured_rag"):
    """Structured RAG function."""
    uri: str = Field(description="The URI of the structured RAG index.")
    get_purchase_history_fn: FunctionRef = Field(default="get_purchase_history", description="The name of the get_purchase_history function.")


@register_function(config_type=StructuredRagFnConfig)
async def structured_rag_fn(
    config: StructuredRagFnConfig, builder: Builder
):
    
    import httpx
    
    get_purchase_history_fn = await builder.get_function(config.get_purchase_history_fn)
        
    structured_rag_search = f"http://{config.uri}/search"

    async def _response_fn(query: str, user_id: str) -> str:

        entry_doc_search = {"query": query, "top_k": 4, "user_id": user_id}
        aggregated_content = ""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(structured_rag_search, json=entry_doc_search)
            # Extract and aggregate the content
            logger.info(f"Actual Structured Response : {response}")
            if response.status_code != 200:
                raise ValueError(f"Error while retireving docs: {response.json()}")
            
            aggregated_content = "\n".join(chunk["content"] for chunk in response.json().get("chunks", []))
            # Check if aggregated_content contains the specific phrase in a case-insensitive manner
            if any(x in aggregated_content.lower() for x in ["no records found", "error:"]):
                raise ValueError("No records found for the specified criteria.")
            return aggregated_content
        except Exception as e:
            logger.info(f"Some error within the structured_rag {e}, sending purchase_history")
            return await get_purchase_history_fn.ainvoke(user_id)

    yield FunctionInfo.create(single_fn=_response_fn)

