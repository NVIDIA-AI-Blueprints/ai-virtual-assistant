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

logger = logging.getLogger(__name__)


class FeedbackResponseFnConfig(FunctionBaseConfig, name="feedback_response"):
    """API for storing useful information for data flywheel."""
    cache_type: str = Field(default="redis",
                            description="The type of the cache to use. Supported types: redis, inmemory")


@register_function(config_type=FeedbackResponseFnConfig)
async def feedback_response_fn(
    config: FeedbackResponseFnConfig, builder: Builder
):

    from aiva_agent.server import FeedbackRequest
    from aiva_agent.server import FeedbackResponse
    from aiva_agent.cache.session_manager import SessionManager

    session_manager = SessionManager(cache_type=config.cache_type)

    async def _response_fn(feedback: FeedbackRequest) -> FeedbackResponse:
        try:
            logger.info(f"Storing user feedback for last response for session {feedback.session_id}")
            session_manager.response_feedback(feedback.session_id, feedback.feedback)
            return FeedbackResponse(message="Response feedback saved successfully")
        except Exception as e:
            logger.error(f"Error in GET /feedback/response endpoint. Error details: {e}")
            return FeedbackResponse(message="Failed to store response feedback")

    yield FunctionInfo.create(single_fn=_response_fn)

