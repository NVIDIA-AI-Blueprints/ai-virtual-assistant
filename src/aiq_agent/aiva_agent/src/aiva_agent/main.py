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
import os
from typing import Annotated, TypedDict, Dict
from langgraph.graph.message import AnyMessage, add_messages
from typing import Callable
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from typing import Annotated, Optional, Literal, TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig

from aiq.profiler.decorators.function_tracking import track_function

from aiva_agent.tools import (
        structured_rag, get_purchase_history, HandleOtherTalk, ProductValidation,
        return_window_validation, update_return, get_recent_return_details,
        ToProductQAAssistant,
        ToOrderStatusAssistant,
        ToReturnProcessing)
from aiva_agent.utils import get_product_name, create_tool_node_with_fallback, canonical_rag
from aiva_agent.utils import get_llm, get_prompts

logger = logging.getLogger(__name__)
prompts = get_prompts()
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
async def handle_other_talk(state: State, config: RunnableConfig):
    """Handles greetings and queries outside order status, returns, or products, providing polite redirection and explaining chatbot limitations."""

    prompt = prompts.get("other_talk_template", "")

    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", prompt),
        ("placeholder", "{messages}"),
        ]
    )

    # LLM
    llm_settings = config.get('configurable', {}).get("llm_settings", default_llm_kwargs)
    llm = get_llm(**llm_settings)
    llm = llm.with_config(tags=["should_stream"])

    # Chain
    small_talk_chain = prompt | llm
    response = await small_talk_chain.ainvoke(state, config)

    return {"messages": [response]}


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


@track_function(metadata={"action": "compute", "source": "api"})
async def ask_clarification(state: State, config: RunnableConfig):

    # Extract the base prompt
    base_prompt = prompts.get("ask_clarification")["base_prompt"]
    previous_conversation = [m for m in state['messages'] if not isinstance(m, ToolMessage)]
    base_prompt = base_prompt.format(previous_conversation=previous_conversation)

    purchase_history = state.get("user_purchase_history", [])
    if state["clarification_type"] == "no_product" and state['reason'].strip():
        followup_prompt = prompts.get("ask_clarification")["followup"]["no_product"].format(
            reason=state['reason'],
            purchase_history=purchase_history
        )
    elif not state['reason'].strip():
        followup_prompt = prompts.get("ask_clarification")["followup"]["default"].format(reason=purchase_history)
    else:
        followup_prompt = prompts.get("ask_clarification")["followup"]["default"].format(reason=state['reason'])

    # Combine base prompt and followup prompt
    prompt = f"{base_prompt} {followup_prompt}"

    # LLM
    llm_settings = config.get('configurable', {}).get("llm_settings", default_llm_kwargs)
    llm = get_llm(**llm_settings)
    llm = llm.with_config(tags=["should_stream"])

    response = await llm.ainvoke(prompt, config)

    return {"messages": [response]}


@track_function(metadata={"action": "compute", "source": "api"})
async def handle_product_qa(state: State, config: RunnableConfig):

    # Extract the previous_conversation
    previous_conversation = [m for m in state['messages'] if not isinstance(m, ToolMessage) and m.content]
    message_type_map = {
        HumanMessage: "user",
        AIMessage: "assistant",
        SystemMessage: "system"
    }

    # Serialized conversation
    get_role = lambda x: message_type_map.get(type(x), None)
    previous_conversation_serialized = [{"role": get_role(m), "content": m.content} for m in previous_conversation if m.content]
    last_message = previous_conversation_serialized[-1]['content']

    retireved_content = canonical_rag(query=last_message, conv_history=previous_conversation_serialized)

    # Use the RAG Template to generate the response
    base_rag_prompt = prompts.get("rag_template")
    rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", base_rag_prompt),
        MessagesPlaceholder("chat_history") + "\n\nCONTEXT:  {context}"
    ]
    )
    rag_prompt = rag_prompt.format(chat_history=previous_conversation, context=retireved_content)

    # LLM
    llm_settings = config.get('configurable', {}).get("llm_settings", default_llm_kwargs)
    llm = get_llm(**llm_settings)
    llm = llm.with_config(tags=["should_stream"])

    response = await llm.ainvoke(rag_prompt, config)

    return {"messages": [response]}


@track_function(metadata={"action": "compute", "source": "api"})
class Assistant:

    def __init__(self, prompt: str, tools: list):
        self.prompt = prompt
        self.tools = tools

    async def __call__(self, state: State, config: RunnableConfig):
        while True:

            llm_settings = config.get('configurable',
                                      {}).get("llm_settings",
                                              default_llm_kwargs)
            llm = get_llm(**llm_settings)
            runnable = self.prompt | llm.bind_tools(self.tools)
            #print("RUNNABLE:", runnable, "\n\n")
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
                result = AIMessage(content="".join(messages))
            else:
                result = await runnable.ainvoke(state)
                #print("LLM RESPONSE:", result, "\n\n")
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

# order status Assistant
order_status_prompt_template = prompts.get("order_status_template", "")

order_status_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            order_status_prompt_template
        ),
        ("placeholder", "{messages}"),
    ]
)

order_status_safe_tools = [structured_rag]
order_status_tools = order_status_safe_tools + [ProductValidation]

# Return Processing Assistant
return_processing_prompt_template = prompts.get("return_processing_template", "")

return_processing_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            return_processing_prompt_template
        ),
        ("placeholder", "{messages}"),
    ]
)

return_processing_safe_tools = [get_recent_return_details, return_window_validation]
return_processing_sensitive_tools = [update_return]
return_processing_tools = return_processing_safe_tools + return_processing_sensitive_tools + [ProductValidation]

primary_assistant_prompt_template = prompts.get("primary_assistant_template", "")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            primary_assistant_prompt_template
        ),
        ("placeholder", "{messages}"),
    ]
)

primary_assistant_tools = [
        HandleOtherTalk,
        ToProductQAAssistant,
        ToOrderStatusAssistant,
        ToReturnProcessing,
    ]

# BUILD THE GRAPH
builder = StateGraph(State)


# SUB AGENTS
# Create product_qa Assistant
builder.add_node(
    "enter_product_qa",
    handle_product_qa,
)

builder.add_edge("enter_product_qa", END)

builder.add_node("order_validation", validate_product_info)
builder.add_node("ask_clarification", ask_clarification)

# Create order_status Assistant
builder.add_node(
    "enter_order_status", create_entry_node("Order Status Assistant")
)
builder.add_node("order_status", Assistant(order_status_prompt, order_status_tools))
builder.add_edge("enter_order_status", "order_status")
builder.add_node(
    "order_status_safe_tools",
    create_tool_node_with_fallback(order_status_safe_tools),
)


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

builder.add_edge("order_status_safe_tools", "order_status")
builder.add_conditional_edges("order_status", route_order_status)

# Create return_processing Assistant
builder.add_node("return_validation", validate_product_info)

builder.add_node(
    "enter_return_processing",
    create_entry_node("Return Processing Assistant"),
)
builder.add_node("return_processing", Assistant(return_processing_prompt, return_processing_tools))
builder.add_edge("enter_return_processing", "return_processing")

builder.add_node(
    "return_processing_safe_tools",
    create_tool_node_with_fallback(return_processing_safe_tools),
)
builder.add_node(
    "return_processing_sensitive_tools",
    create_tool_node_with_fallback(return_processing_sensitive_tools),
)


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


builder.add_edge("return_processing_sensitive_tools", "return_processing")
builder.add_edge("return_processing_safe_tools", "return_processing")
builder.add_conditional_edges("return_processing", route_return_processing)

@track_function(metadata={"action": "compute", "source": "api"})
def user_info(state: State):
    return {"user_purchase_history": get_purchase_history(state["user_id"]), "current_product": ""}

builder.add_node("fetch_purchase_history", user_info)
builder.add_edge(START, "fetch_purchase_history")
builder.add_edge("ask_clarification", END)

# Primary assistant
builder.add_node("primary_assistant", Assistant(primary_assistant_prompt, primary_assistant_tools))
builder.add_node(
    "other_talk", handle_other_talk
)


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
    raise ValueError("Invalid route")

builder.add_edge("other_talk", END)

# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        "enter_product_qa": "enter_product_qa",
        "enter_order_status": "enter_order_status",
        "enter_return_processing": "enter_return_processing",
        "other_talk":"other_talk",
        END: END,
    },
)


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

builder.add_conditional_edges(
    "order_validation",
    is_order_product_valid
)
builder.add_conditional_edges(
    "return_validation",
    is_return_product_valid
)

builder.add_edge("fetch_purchase_history", "primary_assistant")
