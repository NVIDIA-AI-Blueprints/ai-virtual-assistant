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

"""The definition of the Llama Index chain server."""
import os
from uuid import uuid4
import logging
from typing import List
import importlib
import bleach
import time
import prometheus_client
import asyncio
import random
import re
from traceback import print_exc

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel, Field, validator, constr
from aiva_agent.cache.session_manager import SessionManager
from aiva_agent.datastore.datastore import Datastore
from aiva_agent.utils import remove_state_from_checkpointer

from langgraph.errors import GraphRecursionError
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "Health",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {"name": "Feedback", "description": "APIs for storing useful information for data flywheel."},
    {"name": "Session Management", "description": "APIs for managing sessions."},
    {"name": "Inference", "description": "Core APIs for interacting with the agent."},
]


EXAMPLE_DIR = "./"

# List of fallback responses sent out for any Exceptions from /generate endpoint
FALLBACK_RESPONSES = [
    "Please try re-phrasing, I am likely having some trouble with that question.",
    "I will get better with time, please try with a different question.",
    "I wasn't able to process your input. Let's try something else.",
    "Something went wrong. Could you try again in a few seconds with a different question?",
    "Oops, that proved a tad difficult for me, can you retry with another question?"
]

class Message(BaseModel):
    """Definition of the Chat Message type."""
    role: str = Field(description="Role for a message AI, User and System", default="user", max_length=256, pattern=r'[\s\S]*')
    content: str = Field(description="The input query/prompt to the pipeline.", default="Hello what can you do?", max_length=131072, pattern=r'[\s\S]*')

    @validator('role')
    def validate_role(cls, value):
        """ Field validator function to validate values of the field role"""
        value = bleach.clean(value, strip=True)
        valid_roles = {'user', 'assistant', 'system'}
        if value.lower() not in valid_roles:
            raise ValueError("Role must be one of 'user', 'assistant', or 'system'")
        return value.lower()

    @validator('content')
    def sanitize_content(cls, v):
        """ Field validator function to santize user populated feilds from HTML"""
        v = bleach.clean(v, strip=True)
        if not v:  # Check for empty string
            raise ValueError("Message content cannot be empty.")
        return v

class Prompt(BaseModel):
    """Definition of the Prompt API data type."""
    messages: List[Message] = Field(..., description="A list of messages comprising the conversation so far. The roles of the messages must be alternating between user and assistant. The last input message should have role user. A message with the the system role is optional, and must be the very first message if it is present.", max_items=50000)
    user_id: str = Field(None, description="A unique identifier representing your end-user.")
    session_id: str = Field(..., description="A unique identifier representing the session associated with the response.")

class ChainResponseChoices(BaseModel):
    """ Definition of Chain response choices"""
    index: int = Field(default=0, ge=0, le=256, format="int64")
    message: Message = Field(default=Message())
    finish_reason: str = Field(default="", max_length=4096, pattern=r'[\s\S]*')

class ChainResponse(BaseModel):
    """Definition of Chain APIs resopnse data type"""
    id: str = Field(default="", max_length=100000, pattern=r'[\s\S]*')
    choices: List[ChainResponseChoices] = Field(default=[], max_items=256)
    session_id: str = Field(None, description="A unique identifier representing the session associated with the response.")

class DocumentSearch(BaseModel):
    """Definition of the DocumentSearch API data type."""

    query: str = Field(description="The content or keywords to search for within documents.", max_length=131072, pattern=r'[\s\S]*', default="")
    top_k: int = Field(description="The maximum number of documents to return in the response.", default=4, ge=0, le=25, format="int64")

class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""
    content: str = Field(description="The content of the document chunk.", max_length=131072, pattern=r'[\s\S]*', default="")
    filename: str = Field(description="The name of the file the chunk belongs to.", max_length=4096, pattern=r'[\s\S]*', default="")
    score: float = Field(..., description="The relevance score of the chunk.")

class DocumentSearchResponse(BaseModel):
    """Represents a response from a document search."""
    chunks: List[DocumentChunk] = Field(..., description="List of document chunks.", max_items=256)

class DocumentsResponse(BaseModel):
    """Represents the response containing a list of documents."""
    documents: List[constr(max_length=131072, pattern=r'[\s\S]*')] = Field(description="List of filenames.", max_items=1000000, default=[])

class HealthResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

class CreateSessionResponse(BaseModel):
    session_id: str = Field(max_length=4096)

class EndSessionResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

class DeleteSessionResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

class FeedbackRequest(BaseModel):
    """Definition of the Feedback Request data type."""
    feedback: float = Field(..., description="A unique identifier representing your end-user.", ge=-1.0, le=1.0)
    session_id: str = Field(..., description="A unique identifier representing the session associated with the response.")

class FeedbackResponse(BaseModel):
    """Definition of the Feedback Request data type."""
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

# @app.on_event("startup")
# def import_example() -> None:
#     """
#     Import the example class from the specified example file.

#     """

#     file_location = os.path.join(EXAMPLE_DIR, os.environ.get("EXAMPLE_PATH", "basic_rag/llamaindex"))

#     for root, dirs, files in os.walk(file_location):
#         for file in files:
#             if file == "main.py":
#                 # Import the specified file dynamically
#                 spec = importlib.util.spec_from_file_location(name="main", location=os.path.join(root, file))
#                 module = importlib.util.module_from_spec(spec)
#                 spec.loader.exec_module(module)

#                 # Get the Agent app
#                 app.agent = module
#                 break  # Stop the loop once we find and load agent.py

#     # Initialize session manager during startup
#     app.session_manager = SessionManager()

#     # Initialize database to store conversation permanently
#     app.database = Datastore()


def fallback_response_generator(sentence: str, session_id: str = ""):
    """Mock response generator to simulate streaming predefined fallback responses."""

    # Simulate breaking the sentence into chunks (e.g., by word)
    sentence_chunks = sentence.split()  # Split the sentence by words
    resp_id = str(uuid4())  # unique response id for every query
    # Send each chunk (word) in the response
    for chunk in sentence_chunks:
        chain_response = ChainResponse(session_id=session_id, sentiment="")
        response_choice = ChainResponseChoices(
            index=0,
            message=Message(role="assistant", content=f"{chunk} ")
        )
        chain_response.id = resp_id
        chain_response.choices.append(response_choice)
        yield chain_response

    # End with [DONE] response
    chain_response = ChainResponse(session_id=session_id, sentiment="")
    response_choice = ChainResponseChoices(message=Message(role="assistant", content=" "), finish_reason="[DONE]")
    chain_response.id = resp_id
    chain_response.choices.append(response_choice)
    yield chain_response
