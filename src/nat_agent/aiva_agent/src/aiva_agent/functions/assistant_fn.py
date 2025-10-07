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
from nat.data_models.component_ref import LLMRef
from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class AssistantFnConfig(FunctionBaseConfig, name="assistant"):
    """Create a new session."""
    tools: list[FunctionRef] = Field(..., description="The tools to use for the assistant.")
    tool_call_llm: LLMRef = Field(..., description="The tool call LLM to use for the assistant.")
    chat_llm: LLMRef = Field(..., description="The chat LLM to use for the assistant.")
    top_level: bool = Field(..., description="Whether the assistant is the top level assistant.")
    prompt: str = Field(..., description="The prompt to use for the assistant.")
    prompt_config_file: str = Field(default="prompt.yaml", description="The path to the prompt configuration file.")


@register_function(config_type=AssistantFnConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def assistant_fn(fn_config: AssistantFnConfig, builder: Builder):

    from langchain_core.messages import ToolMessage
    from langchain_core.messages import AIMessage
    from langchain_core.prompts.chat import ChatPromptTemplate

    from aiva_agent.state import ctx_routing_level
    from aiva_agent.utils import get_prompts

    # Initialize tools
    tools = await builder.get_tools(fn_config.tools,
                                    wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Initialize LLMs
    tool_call_llm = await builder.get_llm(
        fn_config.tool_call_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tool_call_llm.disable_streaming = True
    tool_call_llm = tool_call_llm.with_config(tags=["should_stream"])

    chat_llm = await builder.get_llm(fn_config.chat_llm,
                                     wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chat_llm.disable_streaming = True
    chat_llm = chat_llm.with_config(tags=["should_stream"])

    # Initialize prompts
    prompts = get_prompts(prompt_config_file=fn_config.prompt_config_file)
    prompt_template = prompts.get(fn_config.prompt)
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("placeholder", "{messages}"),
    ])

    # Initialize runnables
    tool_call_runnable = prompt | tool_call_llm.bind_tools(tools)
    chat_runnable = prompt | chat_llm.bind_tools(tools)

    async def _response_fn(state: dict, config: dict) -> dict:

        routing_level = ctx_routing_level.get()

        while True:
            if routing_level == 0 and fn_config.top_level:
                runnable = tool_call_runnable
            else:
                runnable = chat_runnable
            routing_level += 1

            last_message = state["messages"][-1]
            messages = []

            if isinstance(last_message, ToolMessage) and last_message.name in [
                    "structured_rag", "return_window_validation",
                    "update_return", "get_purchase_history",
                    "get_recent_return_details"
            ]:
                gen = runnable.with_config(
                    tags=["should_stream"],
                    stream_runnable=True,
                    callbacks=config.get(
                        "callbacks",
                        []),  # <-- Propagate callbacks (Python <= 3.10)
                )
                async for message in gen.astream(state):
                    messages.append(message.content)
                result = AIMessage(content="".join(messages)).to_dict()
            else:
                result = await runnable.ainvoke(state)
            if not result.tool_calls and (
                    not result.content or isinstance(result.content, list)
                    and not result.content[0].get("text")):
                messages = state["messages"] + [
                    ("user", "Respond with a real output.")
                ]
                state = {**state, "messages": messages}
                messages = state["messages"] + [
                    ("user", "Respond with a real output.")
                ]
                state = {**state, "messages": messages}
            else:
                break

        return {"messages": result}

    yield FunctionInfo.create(single_fn=_response_fn,
                              description="A tool call assistant.")
