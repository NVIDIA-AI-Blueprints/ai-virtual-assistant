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


class RouteOrderStatusFnConfig(FunctionBaseConfig, name="route_order_status"):
    """Route the primary assistant based on the state."""
    tool_names: list[str] = Field(
        default=["structured_rag", "order_validation"], 
        description="The names of the return processing tools.")


@register_function(config_type=RouteOrderStatusFnConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def route_order_status_fn(
    config: RouteOrderStatusFnConfig, builder: Builder
):
    
    from langgraph.graph import END
    from langgraph.prebuilt import tools_condition
    from aiva_agent.main import State
    from aiva_agent.main import ProductValidation

    async def _response_fn(state: dict) -> str:
        
        state = State(**state)
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        do_product_validation = any(tc["name"] == ProductValidation.__name__ for tc in tool_calls)
        if do_product_validation:
            return "order_validation"
        if all(tc["name"] in config.tool_names for tc in tool_calls):
            return "order_status_safe_tools"
        
        raise ValueError("Invalid route")

    yield FunctionInfo.create(single_fn=_response_fn, description="Route the order status based on the state.")

