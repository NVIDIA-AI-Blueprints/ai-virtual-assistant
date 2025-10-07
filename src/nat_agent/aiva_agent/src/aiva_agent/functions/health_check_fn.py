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


class HealthCheckFnConfig(FunctionBaseConfig, name="health_check"):
    """API for checking and monitoring server liveliness and readiness."""
    response_message: str = Field(default="Service is up.", description="The message to return if the service is healthy")


@register_function(config_type=HealthCheckFnConfig)
async def health_check_fn(
    config: HealthCheckFnConfig, builder: Builder
):

    from pydantic import BaseModel

    class HealthCheckResponse(BaseModel):
        message: str = config.response_message

    async def _response_fn(unused: str | None = None) -> HealthCheckResponse:
        return HealthCheckResponse()

    yield FunctionInfo.create(single_fn=_response_fn)

