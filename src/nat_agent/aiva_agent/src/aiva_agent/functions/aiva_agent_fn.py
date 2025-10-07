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
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.component_ref import FunctionRef

logger = logging.getLogger(__name__)


# TODO: Clean up main/utils, revalidate config/config objects, update dockerfile (local + self contained version)

class AivaAgentFunctionConfig(FunctionBaseConfig, name="aiva_agent"):
    """
    AIQ Toolkit function template. Please update the description.
    """
    visualize_graph: bool = Field(default=False, description="Whether to visualize the graph")
    checkpointer_type: str = Field(default="postgres", description="The type of checkpointer to use")
    checkpointer_url: str = Field(default="postgres:5432", description="The url of the checkpointer")
    checkpointer_db_user: str = Field(default="postgres", description="The user of the checkpointer")
    checkpointer_db_password: str = Field(default="postgres", description="The password of the checkpointer")
    checkpointer_db_name: str = Field(default="postgres", description="The name of the checkpointer")
    cache_type: str = Field(default="redis", description="The type of the cache to use. Supported types: redis, inmemory")
    cache_host: str = Field(default="redis", description="The host of the cache to use. Supported types: redis")
    cache_port: int = Field(default=6379, description="The port of the cache to use. Supported types: redis")
    cache_expiry: int = Field(default=12, description="The expiry of the cache to use. Supported types: redis")
    tool_call_llm_name: str = Field(..., description="The name of the LLM to use for tool call completions.")
    chat_llm_name: str = Field(..., description="The name of the LLM to use for chat completion.")
    handle_product_qa_fn: FunctionRef = Field(
        default="handle_product_qa", description="The name of the tool to use for product QA.")
    handle_other_talk_fn: FunctionRef = Field(
        default="handle_other_talk", description="The name of the tool to use for other talk.")
    ask_clarification_fn: FunctionRef = Field(
        default="ask_clarification", description="The name of the tool to use for ask clarification.")
    validate_product_info_fn: FunctionRef = Field(
        default="validate_product_info", description="The name of the tool to use for validate product info.")
    user_info_fn: FunctionRef = Field(
        default="user_info", description="The name of the tool to use for user info.")
    primary_assistant_fn: FunctionRef = Field(
        default="primary_assistant",
        description="The name of the tool to use for primary assistant.")
    order_status_fn: FunctionRef = Field(
        default="order_status",
        description="The name of the tool to use for the order status assistant.")
    return_processing_fn: FunctionRef = Field(
        default="return_processing",
        description="The name of the tool to use for the return processing assistant.")
    route_primary_assistant_fn: FunctionRef = Field(
        default="route_primary_assistant",
        description="The name of the tool to use for the route primary assistant.")
    route_order_status_fn: FunctionRef = Field(
        default="route_order_status",
        description="The name of the tool to use for the route order status.")
    route_return_processing_fn: FunctionRef = Field(
        default="route_return_processing",
        description="The name of the tool to use for the route return processing.")
    order_status_safe_tools: list[FunctionRef] = Field(
        default=["structured_rag", "ProductValidation"],
        description="The name of the tool to use for the order safe tools.")
    return_processing_safe_tools: list[FunctionRef] = Field(
        default=["get_purchase_history", "return_window_validation"],
        description="The name of the tool to use for the return processing safe tools.")
    return_processing_sensitive_tools: list[FunctionRef] = Field(
        default=["update_return"],
        description="The name of the tool to use for the return processing sensitive tools.")
    is_order_product_valid_fn: FunctionRef = Field(
        default="is_order_product_valid",
        description="The name of the tool to use for the order product valid.")
    is_return_product_valid_fn: FunctionRef = Field(
        default="is_return_product_valid",
        description="The name of the tool to use for the return product valid.")

@register_function(config_type=AivaAgentFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def aiva_agent_function(config: AivaAgentFunctionConfig,
                              builder: Builder):
    import time
    import random
    import re
    import os
    from traceback import print_exc
    from uuid import uuid4
    import asyncio
    from collections.abc import AsyncGenerator

    from langgraph.errors import GraphRecursionError
    from langgraph.graph import START, END
    from langgraph.graph import StateGraph
    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableConfig

    from aiva_agent.cache.session_manager import SessionManager
    from aiva_agent.server import FALLBACK_RESPONSES
    from aiva_agent.server import fallback_response_generator
    from aiva_agent.server import ChainResponse
    from aiva_agent.server import ChainResponseChoices
    from aiva_agent.server import Message
    from aiva_agent.utils import get_checkpointer
    from aiva_agent.server import Prompt
    from aiva_agent.main import State
    from aiva_agent.main import create_entry_node
    from aiva_agent.main import create_tool_node_with_fallback

    # Initialize Tool Call LLM
    tool_call_llm = await builder.get_llm(
        config.tool_call_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tool_call_llm.disable_streaming = True
    tool_call_llm = tool_call_llm.with_config(tags=["should_stream"])

    # Initialize Chat LLM
    chat_llm = await builder.get_llm(config.chat_llm_name,
                                     wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chat_llm.disable_streaming = True
    chat_llm = chat_llm.with_config(tags=["should_stream"])

    # Initialize functions
    handle_product_qa = await builder.get_function(config.handle_product_qa_fn)
    handle_other_talk = await builder.get_function(config.handle_other_talk_fn)
    ask_clarification = await builder.get_function(config.ask_clarification_fn)
    validate_product_info = await builder.get_function(config.validate_product_info_fn)
    user_info = await builder.get_function(config.user_info_fn)
    return_processing = await builder.get_function(config.return_processing_fn)
    route_primary_assistant = await builder.get_function(config.route_primary_assistant_fn)
    route_order_status = await builder.get_function(config.route_order_status_fn)
    route_return_processing = await builder.get_function(config.route_return_processing_fn)
    is_order_product_valid = await builder.get_function(config.is_order_product_valid_fn)
    is_return_product_valid = await builder.get_function(config.is_return_product_valid_fn)

    # Create wrappers as work-around for LangGraph typing enforcement
    _primary_assistant = await builder.get_function(config.primary_assistant_fn)
    _order_status = await builder.get_function(config.order_status_fn)

    async def primary_assistant(state: State, config: RunnableConfig):
        return await _primary_assistant.ainvoke({
            "state": state,
            "config": config
        })

    async def order_status(state: State, config: RunnableConfig):
        return await _order_status.ainvoke({"state": state, "config": config})

    # Initialize tools
    order_status_safe_tools = await builder.get_tools(
        config.order_status_safe_tools,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    return_processing_safe_tools = await builder.get_tools(
        config.return_processing_safe_tools,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    return_processing_sensitive_tools = await builder.get_tools(
        config.return_processing_sensitive_tools,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # BUILD THE GRAPH
    graph_builder = StateGraph(State)

    # Add nodes to the graph
    graph_builder.add_node("enter_product_qa", handle_product_qa.ainvoke)
    graph_builder.add_node("order_validation",
                           validate_product_info.ainvoke)
    graph_builder.add_node("ask_clarification", ask_clarification.ainvoke)
    graph_builder.add_node("enter_order_status",
                           create_entry_node("Order Status Assistant"))
    graph_builder.add_node("order_status", order_status)
    graph_builder.add_node(
        "order_status_safe_tools",
        create_tool_node_with_fallback(order_status_safe_tools))
    graph_builder.add_node("return_validation",
                           validate_product_info.ainvoke)
    graph_builder.add_node("enter_return_processing",
                           create_entry_node("Return Processing Assistant"))
    graph_builder.add_node("return_processing", return_processing.ainvoke)
    graph_builder.add_node(
        "return_processing_safe_tools",
        create_tool_node_with_fallback(return_processing_safe_tools))
    graph_builder.add_node(
        "return_processing_sensitive_tools",
        create_tool_node_with_fallback(return_processing_sensitive_tools))
    graph_builder.add_node("fetch_purchase_history", user_info.ainvoke)
    graph_builder.add_node("primary_assistant", primary_assistant)
    graph_builder.add_node("other_talk", handle_other_talk.ainvoke)

    # Add edges to the graph
    graph_builder.add_edge("enter_product_qa", END)
    graph_builder.add_edge("enter_order_status", "order_status")
    graph_builder.add_edge("enter_return_processing", "return_processing")
    graph_builder.add_conditional_edges(
        "order_status", route_order_status.ainvoke, {
            "order_validation": "order_validation",
            "order_status_safe_tools": "order_status_safe_tools",
            END: END
        })
    graph_builder.add_edge("order_status_safe_tools", "order_status")
    graph_builder.add_edge("return_processing_sensitive_tools",
                           "return_processing")
    graph_builder.add_edge("return_processing_safe_tools", "return_processing")
    graph_builder.add_conditional_edges(
        "return_processing", route_return_processing.ainvoke, {
            "return_validation": "return_validation",
            "return_processing_safe_tools": "return_processing_safe_tools",
            "return_processing_sensitive_tools":
            "return_processing_sensitive_tools",
            END: END
        })
    graph_builder.add_edge(START, "fetch_purchase_history")
    graph_builder.add_edge("ask_clarification", END)
    graph_builder.add_edge("other_talk", END)
    graph_builder.add_conditional_edges(
        "primary_assistant",
        route_primary_assistant.ainvoke,
        {
            "enter_product_qa": "enter_product_qa",
            "enter_order_status": "enter_order_status",
            "enter_return_processing": "enter_return_processing",
            "other_talk": "other_talk",
            END: END,
        },
    )
    graph_builder.add_conditional_edges("order_validation",
                                        is_order_product_valid.ainvoke)
    graph_builder.add_conditional_edges("return_validation",
                                        is_return_product_valid.ainvoke)
    graph_builder.add_edge("fetch_purchase_history", "primary_assistant")

    # Compile graph with checkpointer
    # memory, pool = await get_checkpointer(
    #     checkpointer_type=config.checkpointer_type,
    #     url=config.checkpointer_url,
    #     db_user=config.checkpointer_db_user,
    #     db_password=config.checkpointer_db_password,
    #     db_name=config.checkpointer_db_name)

    # Compile graph with inmemory checkpointer (works perfectly!)
    memory, pool = await get_checkpointer(checkpointer_type="inmemory")

    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["return_processing_sensitive_tools"])

    # Visualize the graph
    if config.visualize_graph:
        try:
            # Generate the PNG image from the graph
            png_image_data = graph.get_graph(xray=True).draw_mermaid_png()
            # Save the image to a file in the current directory
            with open("graph_image.png", "wb") as f:
                f.write(png_image_data)
        except Exception as e:
            # This requires some extra dependencies and is optional
            logger.info(f"An error occurred: {e}")

    # Initialize session manager
    session_manager = SessionManager(cache_type=config.cache_type,
                                     host=config.cache_host,
                                     port=config.cache_port,
                                     expiry=config.cache_expiry)

    # Implement your function logic here
    async def _response_stream_fn(
            prompt: Prompt) -> AsyncGenerator[ChainResponse, None]:
        # Process the input_message and generate output
        user_query_timestamp = time.time()

        # Handle invalid session id
        if not session_manager.is_session(prompt.session_id):  # TODO FIX THIS
            logger.error(
                f"No session_id created {prompt.session_id}. Please create session id before generate request."
            )
            print_exc()
            for data in fallback_response_generator(
                    sentence=random.choice(FALLBACK_RESPONSES),
                    session_id=prompt.session_id):
                yield data
            return

        chat_history = prompt.messages

        # The last user message will be the query for the rag or llm chain
        last_user_message = next(
            (message.content
             for message in reversed(chat_history) if message.role == 'user'),
            None)

        # Normalize the last user input and remove non-ascii characters
        last_user_message = re.sub(
            r'[^\x00-\x7F]+', '',
            last_user_message)  # Remove all non-ascii characters
        last_user_message = re.sub(
            r'[\u2122\u00AE]', '', last_user_message
        )  # Remove standard trademark and copyright symbols
        last_user_message = last_user_message.replace("~", "-")
        logger.info(f"Normalized user input: {last_user_message}")

        # Keep copy of unmodified query to store in db
        user_query = last_user_message

        log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
        debug_langgraph = False
        if log_level == "DEBUG":
            debug_langgraph = True

        recursion_limit = int(os.environ.get('GRAPH_RECURSION_LIMIT', '6'))

        async def response_generator() -> AsyncGenerator[ChainResponse, None]:

            try:
                resp_id = str(uuid4())
                is_exception = False
                # Variable to track if this is the first yield
                is_first_yield = True
                resp_str = ""
                last_content = ""

                logger.info(
                    f"Chat History:  {session_manager.get_conversation(prompt.session_id)}"
                )
                config = {
                    "recursion_limit": recursion_limit,
                    "configurable": {
                        "thread_id":
                        prompt.session_id,
                        "chat_history":
                        session_manager.get_conversation(prompt.session_id)
                    }
                }

                # Check for the interrupt
                snapshot = await graph.aget_state(config)

                if not snapshot.next:
                    input_for_graph = {
                        "messages": [("human", last_user_message)],
                        "user_id": prompt.user_id
                    }
                else:
                    if last_user_message.strip().startswith(
                        ("Yes", "yes", "Y", "y")):
                        # Just continue
                        input_for_graph = None
                    else:
                        last_item = snapshot.values.get("messages")[-1]
                        if last_item and hasattr(
                                last_item,
                                "tool_calls") and last_item.tool_calls:
                            input_for_graph = {
                                "messages": [
                                    ToolMessage(
                                        tool_call_id=last_item.tool_calls[0]
                                        ["id"],
                                        content=
                                        f"API call denied by user. Reasoning: '{last_user_message}'. Continue assisting, accounting for the user's input.",
                                    )
                                ]
                            }
                        elif not hasattr(last_item, "tool_calls"):
                            input_for_graph = {
                                "messages": [("human", last_user_message)],
                                "user_id": prompt.user_id
                            }
                        else:
                            input_for_graph = None

                try:
                    function_start_time = time.time()
                    # Set Maximum time to wait for a step to complete, in seconds. Defaults to None
                    graph_timeout_env = os.environ.get('GRAPH_TIMEOUT_IN_SEC',
                                                       None)
                    graph.step_timeout = int(
                        graph_timeout_env) if graph_timeout_env else None
                    async for event in graph.astream_events(
                            input_for_graph,
                            version="v2",
                            config=config,
                            debug=debug_langgraph):
                        kind = event["event"]
                        tags = event.get("tags", [])
                        if kind == "on_chain_end" and event['data'].get(
                                'output', "") == '__end__':
                            end_msgs = event['data']['input']['messages']
                            last_content = end_msgs[-1].content
                            print("last_content: ", last_content)
                        if kind == "on_chat_model_stream" and "should_stream" in tags:
                            content = event["data"]["chunk"].content
                            resp_str += content
                            if content:
                                chain_response = ChainResponse()
                                response_choice = ChainResponseChoices(
                                    index=0,
                                    message=Message(role="assistant",
                                                    content=content))
                                chain_response.id = resp_id
                                chain_response.session_id = prompt.session_id
                                chain_response.choices.append(response_choice)
                                logger.debug(response_choice)
                                # Check if this is the first yield
                                if is_first_yield:
                                    logger.info(
                                        f"Execution time until first yield:  {time.time() - function_start_time}"
                                    )
                                    is_first_yield = False

                                yield chain_response
                    # If resp_str is empty after the loop, use the last AI message content
                    # If there is no Streaming response
                    if not resp_str and last_content:
                        chain_response = ChainResponse()
                        response_choice = ChainResponseChoices(
                            index=0,
                            message=Message(role="assistant",
                                            content=last_content))
                        chain_response.id = resp_id
                        chain_response.session_id = prompt.session_id
                        chain_response.choices.append(response_choice)
                        yield chain_response
                        resp_str = last_content
                        logger.debug(
                            f"Using last AI message content as the final response: {last_content}"
                        )

                    snapshot = await graph.aget_state(config)
                    # If there is a snapshot ask the user for return confirmation
                    if snapshot.next:
                        user_confirmation = "Do you approve of the process the return? Type 'y' to continue; otherwise, explain your requested changed."
                        chain_response = ChainResponse()
                        response_choice = ChainResponseChoices(
                            index=0,
                            message=Message(role="assistant",
                                            content=user_confirmation))
                        chain_response.id = resp_id
                        chain_response.session_id = prompt.session_id
                        chain_response.choices.append(response_choice)
                        logger.debug(response_choice)
                        yield chain_response
                # Check for the interrupt
                except asyncio.TimeoutError as te:
                    logger.info(
                        "This issue may occur if the LLM takes longer to respond. The timeout duration can be configured using the environment variable GRAPH_TIMEOUT_IN_SEC."
                    )
                    logger.error(f"Graph Timeout Error. Error details: {te}")
                    is_exception = True
                except GraphRecursionError as ge:
                    logger.error(f"Graph Recursion Error. Error details: {ge}")
                    is_exception = True

            except AttributeError as attr_err:
                # Catch any specific attribute errors and log them
                logger.error(f"AttributeError: {attr_err}")
                print_exc()
                is_exception = True
            except asyncio.CancelledError as e:
                logger.error(f"Task was cancelled. Details: {e}")
                print_exc()
                is_exception = True
            except Exception as e:
                logger.error(
                    f"Sending empty response. Unexpected error in response_generator: {e}"
                )
                print_exc()
                is_exception = True

            if is_exception:
                logger.error(
                    "Sending back fallback responses since an exception was raised."
                )
                is_exception = False
                for data in fallback_response_generator(
                        sentence=random.choice(FALLBACK_RESPONSES),
                        session_id=prompt.session_id):
                    yield data

            chain_response = ChainResponse()
            # Initialize content with space to overwrite default response
            response_choice = ChainResponseChoices(index=0,
                                                   message=Message(
                                                       role="assistant",
                                                       content=' '),
                                                   finish_reason="[DONE]")

            logger.info(
                f"Conversation saved:\nSession ID: {prompt.session_id}\nQuery: {last_user_message}\nResponse: {resp_str}"
            )
            session_manager.save_conversation(
                prompt.session_id,
                prompt.user_id or "",
                [
                    {
                        "role": "user",
                        "content": user_query,
                        "timestamp": f"{user_query_timestamp}"
                    },
                    {
                        "role": "assistant",
                        "content": resp_str,
                        "timestamp": f"{time.time()}"
                    },
                ],
            )

            chain_response.id = resp_id
            chain_response.session_id = prompt.session_id
            chain_response.choices.append(response_choice)
            logger.debug(response_choice)

            yield chain_response

        async for chunk in response_generator():
            yield chunk

    try:
        yield FunctionInfo.create(stream_fn=_response_stream_fn,
                                  description="AIVA Agent")
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up aiva_agent workflow.")
