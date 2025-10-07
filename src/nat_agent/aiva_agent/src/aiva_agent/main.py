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
from typing import Annotated, TypedDict, Dict
from langgraph.graph.message import AnyMessage, add_messages
from typing import Callable
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from typing import Annotated, Optional, Literal, TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel

from nat.profiler.decorators.function_tracking import track_function

from aiva_agent.tools import (
        structured_rag, get_purchase_history, HandleOtherTalk, ProductValidation,
        return_window_validation, update_return, get_recent_return_details,
        ToProductQAAssistant,
        ToOrderStatusAssistant,
        ToReturnProcessing)
from aiva_agent.utils import create_tool_node_with_fallback

logger = logging.getLogger(__name__)
# TODO get the default_kwargs from the Agent Server API
default_llm_kwargs = {"temperature": 0.2, "top_p": 0.7, "max_tokens": 1024}

# STATE OF THE AGENT
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    user_purchase_history: Dict
    current_product: str
    needs_clarification: bool
    clarification_type: str
    reason: str


# NODES FOR THE AGENT
@track_function(metadata={"action": "compute", "source": "api"})
def validate_product_info(state: State, config: RunnableConfig):
    # This node will take user history and find product name based on query
    # If there are multiple name of no name specified in the graph then it will

    # This dict is to populate the user_purchase_history and product details if required
    response_dict = {"needs_clarification": False}
    if state["user_id"]:
        # Update user purchase history based
        response_dict.update({"user_purchase_history": get_purchase_history(state["user_id"])})

        # Extracting product name which user is expecting
        product_list = list(set([resp.get("product_name") for resp in response_dict.get("user_purchase_history", [])]))

        # Extract product name from query and filter from database
        product_info = get_product_name(state["messages"], product_list)

        product_names = product_info.get("products_from_purchase", [])
        product_in_query = product_info.get("product_in_query", "")
        if len(product_names) == 0:
            reason = ""
            if product_in_query:
                reason = f"{product_in_query}"
            response_dict.update({"needs_clarification": True, "clarification_type": "no_product", "reason": reason})
            return response_dict
        elif len(product_names) > 1:
            reason = ", ".join(product_names)
            response_dict.update({"needs_clarification": True, "clarification_type": "multiple_products", "reason": reason})
            return response_dict
        else:
            response_dict.update({"current_product": product_names[0]})

    return response_dict


@track_function(metadata={"action": "compute", "source": "api"})
def create_entry_node(assistant_name: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ]
        }

    return entry_node


# @track_function(metadata={"action": "compute", "source": "api"})
# class Assistant:

#     def __init__(self, prompt: str, tools: list, tool_call_llm: BaseChatModel, chat_llm: BaseChatModel, top_level=False):
#         self.prompt = prompt
#         self.tools = tools
#         self.tool_call_llm = tool_call_llm
#         self.chat_llm = chat_llm
#         self.top_level = top_level
    
#     async def __call__(self, state: State, config: RunnableConfig):

#         routing_level = ctx_routing_level.get()
        
#         while True:

#             # Workaround ensuring the tool call model is only used the first time
#             # graph execution
#             if routing_level == 0 and self.top_level:
#                 llm = self.tool_call_llm
#             else:
#                 llm = self.chat_llm
#             routing_level += 1

#             llm_settings = config.get('configurable',
#                                      {}).get("llm_settings",
#                                              default_llm_kwargs)

#             runnable = self.prompt | llm.bind_tools(self.tools)
#             last_message = state["messages"][-1]
#             messages = []
#             if isinstance(last_message, ToolMessage) and last_message.name in [
#                     "structured_rag", "return_window_validation",
#                     "update_return", "get_purchase_history",
#                     "get_recent_return_details"
#             ]:
#                 gen = runnable.with_config(
#                     tags=["should_stream"],
#                     stream_runnable=True,
#                     callbacks=config.get(
#                         "callbacks",
#                         []),  # <-- Propagate callbacks (Python <= 3.10)
#                 )
#                 async for message in gen.astream(state):
#                     messages.append(message.content)
#                 result = AIMessage(content="".join(messages))
#             else:
#                 result = await runnable.ainvoke(state)
#             if not result.tool_calls and (
#                     not result.content or isinstance(result.content, list)
#                     and not result.content[0].get("text")):
#                 messages = state["messages"] + [
#                     ("user", "Respond with a real output.")
#                 ]
#                 state = {**state, "messages": messages}
#                 messages = state["messages"] + [
#                     ("user", "Respond with a real output.")
#                 ]
#                 state = {**state, "messages": messages}
#             else:
#                 break
#         return {"messages": result}


def create_prompt(prompts: dict, prompt_name: str) -> ChatPromptTemplate:
    prompt_template = prompts.get(prompt_name, None)
    if not prompt_template:
        raise ValueError(f"Prompt template {prompt_name} not found in prompts")
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("placeholder", "{messages}"),
        ]
    )


# order status Assistant
order_status_safe_tools = [structured_rag]
order_status_tools = order_status_safe_tools + [ProductValidation]


# Return Processing Assistant
return_processing_safe_tools = [get_recent_return_details, return_window_validation]
return_processing_sensitive_tools = [update_return]
return_processing_tools = return_processing_safe_tools + return_processing_sensitive_tools + [ProductValidation]


# Primary Assistant
primary_assistant_tools = [
        HandleOtherTalk,
        ToProductQAAssistant,
        ToOrderStatusAssistant,
        ToReturnProcessing,
    ]


@track_function(metadata={"action": "compute", "source": "api"})
def route_order_status(
    state: State,
) -> Literal[
    "order_status_safe_tools",
    "order_validation",
    "__end__"
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    tool_names = [t.name for t in order_status_safe_tools]
    do_product_validation = any(tc["name"] == ProductValidation.__name__ for tc in tool_calls)
    if do_product_validation:
        return "order_validation"
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "order_status_safe_tools"
    return "order_status_sensitive_tools"


@track_function(metadata={"action": "compute", "source": "api"})
def route_return_processing(
    state: State,
) -> Literal[
    "return_processing_safe_tools",
    "return_processing_sensitive_tools",
    "return_validation",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    do_product_validation = any(tc["name"] == ProductValidation.__name__ for tc in tool_calls)
    if do_product_validation:
        return "return_validation"
    tool_names = [t.name for t in return_processing_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "return_processing_safe_tools"
    return "return_processing_sensitive_tools"


@track_function(metadata={"action": "compute", "source": "api"})
def user_info(state: State):
    return {"user_purchase_history": get_purchase_history(state["user_id"]), "current_product": ""}


#  Add "primary_assistant_tools", if necessary
@track_function(metadata={"action": "compute", "source": "api"})
def route_primary_assistant(
    state: State,
) -> Literal[
    "enter_product_qa",
    "enter_order_status",
    "enter_return_processing",
    "other_talk",
    "__end__",
]:
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
    
    print("TOOL_CALLS: ", tool_calls)
    print("Test__name__: ", ToReturnProcessing.__name__)

    raise ValueError("Invalid route")


@track_function(metadata={"action": "compute", "source": "api"})
def is_order_product_valid(state: State)  -> Literal[
    "ask_clarification",
    "order_status"
]:
    """Conditional edge from validation node to decide if we should ask followup questions"""
    if state["needs_clarification"] == True:
        return "ask_clarification"
    return "order_status"


@track_function(metadata={"action": "compute", "source": "api"})
def is_return_product_valid(state: State)  -> Literal[
    "ask_clarification",
    "return_processing"
]:
    """Conditional edge from validation node to decide if we should ask followup questions"""
    if state["needs_clarification"] == True:
        return "ask_clarification"
    return "return_processing"
