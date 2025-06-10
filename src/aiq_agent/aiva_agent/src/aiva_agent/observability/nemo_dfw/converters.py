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
import json
import typing

from opentelemetry.sdk.trace import ReadableSpan

from aiva_agent.observability.nemo_dfw.schemas.dfw_record import Request
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import RequestTool
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import Function
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import MessageToolCall
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import ToolMessage
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import AssistantMessage
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import FunctionMessage
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import Response
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import ChoiceMessageToolCall
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import ChoiceMessageToolCallFunction
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import ChoiceResponseMessage
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import ResponseChoice
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import Role
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import FinishReason
from aiva_agent.observability.nemo_dfw.schemas.dfw_record import NemoDFWRecord

logger = logging.getLogger(__name__)


def _get_structured_attribute(span: ReadableSpan, 
                              attribute_name: str, 
                              default_value: typing.Any = "{}") -> dict | list | None:
    """Get a structured attribute from a span

    Args:
        span (ReadableSpan): The span to get the attribute from
        attribute_name (str): The name of the attribute to get
        default_value (typing.Any): The default value to return if the attribute is not found
    """

    try:
        serialized_attribute = span.attributes.get(attribute_name, default_value)
        deserialized_attribute = json.loads(serialized_attribute)
        return deserialized_attribute
    except json.JSONDecodeError:
        logger.error("Invalid JSON in %s: %s", attribute_name, serialized_attribute)

    return
   

def _convert_chat_response(chat_response: dict) -> ResponseChoice | None:
    """Convert a chat response to a DFW payload

    Args:
        chat_response (dict): The chat response to convert

    Returns:
        ChatCompletionResponse | None: The converted chat response
    """    
    message = chat_response.get("message", {})
    if message is not None:
        # Get content
        content = message.get("content", "")
        # Get role
        response_message = message.get("response_metadata", {})
        finish_reason = response_message.get("finish_reason", {})
        # DFW only supports tool calls
        if finish_reason != FinishReason.TOOL_CALLS.value:
            return
        if response_message is not None:
            role = response_message.get("role", "") # todo think about default role
            role = _convert_role(role)
        # Get tool calls
        validated_tool_calls = []
        additional_kwargs = message.get("additional_kwargs", {})
        if additional_kwargs is not None:
            tool_calls = additional_kwargs.get("tool_calls", [])
            if tool_calls is not None:
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    function_args = json.loads(function.get("arguments", {})) or {}
                    validated_tool_calls.append(ChoiceMessageToolCall(
                        type= tool_call.get("type", "function"), 
                        function=ChoiceMessageToolCallFunction(
                            id=tool_call.get("id", ""),
                            name=function.get("name", "unknown") or "unknown",
                            arguments=function_args)))
        response_choice = ResponseChoice(
            message=ChoiceResponseMessage(
                content=content,
                role=role,
                tool_calls=validated_tool_calls)) 

        return response_choice       


RoleMap = {
    "human": Role.USER.value,
    "user": Role.USER.value,
    "assistant": Role.ASSISTANT.value,
    "system": Role.SYSTEM.value,
    "ai": Role.AI.value,
    "tool": Role.TOOL.value
}

def _convert_role(role: str) -> str:
    return RoleMap.get(role, "unknown")

def _convert_other_input_message(message: dict) -> dict:
    """Convert a other input message to a DFW payload

    Args:
        message (dict): The message to convert

    Returns:
        dict: The converted message
    """
    converted_message = {}
    
    response_metadata = message.get("response_metadata", {})
    # Get content -- Tool call message content must be None
    if "content" in response_metadata:
        converted_message["content"] = response_metadata.get("content", None)
    else:
        converted_message["content"] = message.get("content", "") or ""  

    # Get role
    role = response_metadata.get("role", message.get("type"))
    role = _convert_role(role)

    converted_message["role"] = role

    # Get name
    converted_message["name"] = response_metadata.get("name", "")
    # Get tool calls
    additional_kwargs = message.get("additional_kwargs", {})
    validated_tool_calls = []
    if additional_kwargs is not None:
        tool_calls = additional_kwargs.get("tool_calls", [])
        if tool_calls is not None:
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                function_args = json.loads(function.get("arguments", {})) or {}
                tool_call_obj = MessageToolCall(
                    id=tool_call.get("id", ""),
                    type= tool_call.get("type", "function"), 
                    function=Function(
                        name=function.get("name", "unknown") or "unknown",
                        arguments=function_args)) 
                validated_tool_calls.append(tool_call_obj) 
        converted_message["tool_calls"] = validated_tool_calls   

    return converted_message


def _convert_tool_input_message(message: dict) -> dict:
    """Convert a tool input message to a DFW payload

    Args:
        message (dict): The message to convert

    Returns:
        dict: The converted message
    """

    converted_message = {}

    response_metadata = message.get("response_metadata", {})
    # Get content -- Tool call message content must be None
    if "content" in response_metadata:
        converted_message["content"] = response_metadata.get("content", None)
    else:
        converted_message["content"] = message.get("content", "") or ""   
    # Get role
    role = response_metadata.get("role", message.get("type"))
    role = _convert_role(role)

    converted_message["role"] = role 
    converted_message["tool_call_id"] = message.get("tool_call_id", "")   

    return converted_message

def otel_span_to_dfw_record(span: ReadableSpan) -> NemoDFWRecord:
    """Convert a span to a DFW payload

    Args:
        span (ReadableSpan): A span from the OpenTelemetry SDK

    Returns:
        NemoDFWRecord: The converted DFW payload that is compatible with the NeMo Data Flywheel
    """

    # Only proceed if there are valid span attributes
    if span.attributes is None:
        return

    # Transform request messages
    message_content_list: list | None = _get_structured_attribute(span, "input.value")

    # Only proceed if there are valid messages
    if message_content_list is None:
        return    

    # Transform request messages
    messages = []
    for message in message_content_list:

        message_type = message.get("type", None)

        match message_type:
            case "user" | "system" | "assistant" | "human" | "ai":
                msg_result = _convert_other_input_message(message)
            case "tool":
                msg_result = _convert_tool_input_message(message)
            case _:
                return
            
        messages.append(msg_result)

    metadata: dict = _get_structured_attribute(span, "aiq.metadata") or {}
    tools = metadata.get("tools_schema", [])

    # Construct a Request object
    request = Request(messages=messages, model=str(span.attributes.get("aiq.subspan.name", "")), tools=tools)  

    # Until DFW only supports more than system/human messages
    types_to_check = (ToolMessage, AssistantMessage, FunctionMessage)
    if any(isinstance(model, tuple(types_to_check)) for model in request.messages):
        return
    
    # Transform response messages
    response_choices = []
    chat_responses = metadata.get("chat_responses", []) or []
    for chat_response in chat_responses:
        response_choice = _convert_chat_response(chat_response)

        # DFW only supports tool calls
        if response_choice is None:
            return
        
        response_choices.append(response_choice)
    
    responses = Response(choices=response_choices) 
    
    dfw_payload = NemoDFWRecord(request=request, 
                                response=responses, 
                                timestamp=int(float(str(span.attributes.get("aiq.event_timestamp", 0)))),
                                workload_id=span.attributes.get("aiq.function.name", "unknown"))

    return dfw_payload
