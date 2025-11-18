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
from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class RoutePrimaryAsisstantFnConfig(FunctionBaseConfig, name="route_primary_assistant"):
    """Route the primary assistant based on the state."""
    pass


@register_function(config_type=RoutePrimaryAsisstantFnConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def route_primary_assistant_fn(
    config: RoutePrimaryAsisstantFnConfig, builder: Builder
):
    
    from langgraph.graph import END
    from langgraph.prebuilt import tools_condition
    from aiva_agent.main import State
    from aiva_agent.main import ToProductQAAssistant
    from aiva_agent.main import ToOrderStatusAssistant
    from aiva_agent.main import ToReturnProcessing
    from aiva_agent.main import HandleOtherTalk

    async def _response_fn(state: dict) -> str:

        state = State(**state)
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        if tool_calls:
            if tool_calls[0]["name"] == ToProductQAAssistant.__name__:
                return "enter_product_qa"
            elif tool_calls[0]["name"] == ToOrderStatusAssistant.__name__:
                return "enter_order_status"
            elif tool_calls[0]["name"] == ToReturnProcessing.__name__:
                return "enter_return_processing"
            elif tool_calls[0]["name"] == HandleOtherTalk.__name__:
                return "other_talk"

        raise ValueError("Invalid route")

    yield FunctionInfo.create(single_fn=_response_fn, description="Route the primary assistant based on the state.")

