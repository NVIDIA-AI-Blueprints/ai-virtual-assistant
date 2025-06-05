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

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class AivaAgentFunctionConfig(FunctionBaseConfig, name="aiva_agent"):
    """
    AIQ Toolkit function template. Please update the description.
    """
    visualize_graph: bool = Field(default=False, description="Whether to visualize the graph")


@register_function(config_type=AivaAgentFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def aiva_agent_function(
    config: AivaAgentFunctionConfig, builder: Builder
):

    import time
    import random
    import re
    import os
    from traceback import print_exc
    from uuid import uuid4
    import asyncio
    from collections.abc import AsyncGenerator

    from langgraph.errors import GraphRecursionError
    from langchain_core.messages import ToolMessage

    from aiva_agent.cache.session_manager import SessionManager
    from aiva_agent.server import FALLBACK_RESPONSES
    from aiva_agent.server import fallback_response_generator
    from aiva_agent.server import StreamingResponse
    from aiva_agent.server import ChainResponse
    from aiva_agent.server import ChainResponseChoices
    from aiva_agent.server import Message 
    from aiva_agent.utils import get_checkpointer
    from aiva_agent.main import builder       
    from aiva_agent.server import Prompt

    memory, pool = await get_checkpointer()
    
    # Compile graph with checkpointer
    graph = builder.compile(checkpointer=memory,
                            interrupt_before=["return_processing_sensitive_tools"])
    
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
    
    session_manager = SessionManager()
    
    # Implement your function logic here
    async def _response_stream_fn(prompt: Prompt) -> AsyncGenerator[ChainResponse, None]:
        # Process the input_message and generate output
        user_query_timestamp = time.time()

        # Handle invalid session id
        if not session_manager.is_session(prompt.session_id):  # TODO FIX THIS
            logger.error(f"No session_id created {prompt.session_id}. Please create session id before generate request.")
            print_exc()
            for data in fallback_response_generator(sentence=random.choice(FALLBACK_RESPONSES), session_id=prompt.session_id):
                yield data
            return
        
        chat_history = prompt.messages

        # The last user message will be the query for the rag or llm chain
        last_user_message = next((message.content for message in reversed(chat_history) if message.role == 'user'), None)

        # Normalize the last user input and remove non-ascii characters
        last_user_message = re.sub(r'[^\x00-\x7F]+', '', last_user_message) # Remove all non-ascii characters
        last_user_message = re.sub(r'[\u2122\u00AE]', '', last_user_message) # Remove standard trademark and copyright symbols
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

                logger.info(f"Chat History:  {session_manager.get_conversation(prompt.session_id)}")
                config = {"recursion_limit": recursion_limit,
                    "configurable": {"thread_id": prompt.session_id, "chat_history": session_manager.get_conversation(prompt.session_id)}} 

                # Check for the interrupt
                snapshot = await graph.aget_state(config)

                if not snapshot.next:
                    input_for_graph = {"messages":[("human", last_user_message)], "user_id": prompt.user_id}   
                else:
                    if last_user_message.strip().startswith(("Yes", "yes", "Y", "y")):
                        # Just continue
                        input_for_graph = None
                    else:
                        last_item = snapshot.values.get("messages")[-1]
                        if last_item and hasattr(last_item, "tool_calls") and last_item.tool_calls:
                            input_for_graph = {
                                        "messages": [
                                            ToolMessage(
                                                tool_call_id=last_item.tool_calls[0]["id"],
                                                content=f"API call denied by user. Reasoning: '{last_user_message}'. Continue assisting, accounting for the user's input.",
                                            )
                                        ]
                                    }
                        elif not hasattr(last_item, "tool_calls"):
                            input_for_graph = {"messages":[("human", last_user_message)], "user_id": prompt.user_id}
                        else:
                            input_for_graph = None
                
                try:
                    function_start_time = time.time()
                    # Set Maximum time to wait for a step to complete, in seconds. Defaults to None
                    graph_timeout_env =  os.environ.get('GRAPH_TIMEOUT_IN_SEC', None)           
                    #agent.graph.step_timeout = int(graph_timeout_env) if graph_timeout_env else None  
                    graph.step_timeout = int(graph_timeout_env) if graph_timeout_env else None  
                    async for event in graph.astream_events(input_for_graph, version="v2", config=config, debug=debug_langgraph): # TODO FIX BUG IN LANGCHAIN
                    #async for event in graph.astream_events(input_for_graph, version="v2", config=config, debug=False): 
                        #print(event)
                        kind = event["event"]
                        tags = event.get("tags", [])
                        if kind == "on_chain_end" and event['data'].get('output', "") == '__end__':
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
                                    message=Message(
                                        role="assistant",
                                        content=content
                                    )
                                )  
                                chain_response.id = resp_id
                                chain_response.session_id = prompt.session_id
                                chain_response.choices.append(response_choice)
                                logger.debug(response_choice)
                                # Check if this is the first yield
                                if is_first_yield:
                                    logger.info(f"Execution time until first yield:  {time.time() - function_start_time}")
                                    is_first_yield = False

                                yield chain_response
                    # If resp_str is empty after the loop, use the last AI message content
                    # If there is no Streaming response
                    if not resp_str and last_content:
                        chain_response = ChainResponse()
                        response_choice = ChainResponseChoices(
                            index=0,
                            message=Message(
                                role="assistant",
                                content=last_content
                            )
                        )
                        chain_response.id = resp_id
                        chain_response.session_id = prompt.session_id
                        chain_response.choices.append(response_choice)
                        yield chain_response
                        resp_str = last_content
                        logger.debug(f"Using last AI message content as the final response: {last_content}")   
                    
                    snapshot = await graph.aget_state(config)
                    # If there is a snapshot ask the user for return confirmation
                    if snapshot.next:
                        user_confirmation = "Do you approve of the process the return? Type 'y' to continue; otherwise, explain your requested changed."
                        chain_response = ChainResponse()
                        response_choice = ChainResponseChoices(
                            index=0,
                            message=Message(
                                role="assistant",
                                content=user_confirmation
                            )
                        )
                        chain_response.id = resp_id
                        chain_response.session_id = prompt.session_id
                        chain_response.choices.append(response_choice)
                        logger.debug(response_choice)
                        yield chain_response
                # Check for the interrupt
                except asyncio.TimeoutError as te:
                    logger.info("This issue may occur if the LLM takes longer to respond. The timeout duration can be configured using the environment variable GRAPH_TIMEOUT_IN_SEC.")
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
                logger.error(f"Sending empty response. Unexpected error in response_generator: {e}")
                print_exc()
                is_exception = True

            if is_exception:
                logger.error("Sending back fallback responses since an exception was raised.")
                is_exception = False
                for data in fallback_response_generator(sentence=random.choice(FALLBACK_RESPONSES), session_id=prompt.session_id):
                    yield data       

            chain_response = ChainResponse()
            # Initialize content with space to overwrite default response
            response_choice = ChainResponseChoices(
                        index=0,
                        message=Message(
                            role="assistant",
                            content=' '
                        ),
                        finish_reason="[DONE]"
                    )        

            logger.info(f"Conversation saved:\nSession ID: {prompt.session_id}\nQuery: {last_user_message}\nResponse: {resp_str}")
            session_manager.save_conversation(
                prompt.session_id,
                prompt.user_id or "",
                [
                    {"role": "user", "content": user_query, "timestamp": f"{user_query_timestamp}"},
                    {"role": "assistant", "content": resp_str, "timestamp": f"{time.time()}"},
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
        yield FunctionInfo.create(stream_fn=_response_stream_fn, description="AIVA Agent")
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up aiva_agent workflow.")