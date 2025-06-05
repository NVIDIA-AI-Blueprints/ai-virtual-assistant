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


import os
import logging
from typing import List
import bleach
import prometheus_client
from uuid import uuid4
import httpx
import asyncio

# For session response
from fastapi import Response
from pydantic import BaseModel
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from traceback import print_exc


logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "Health",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {"name": "Agent", "description": "Core APIs for interacting with the agent."}
]


# create the FastAPI server
app = FastAPI(title="API Gateway server for AI Virtual Assistant for Customer Service",
    description="This API schema describes all the endpoints exposed by the AI Virtual Assistant for Customer Service NIM Blueprint",
    version="1.0.0",
        docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
)

# Allow access in browser from RAG UI and Storybook (development)
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

AGENT_SERVER_URL = os.getenv("AGENT_SERVER_URL")
ANALYTICS_SERVER_URL = os.getenv("ANALYTICS_SERVER_URL")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 180))

class HealthResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")

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
    def sanitize_content(cls, v, values):
        """Field validator function to sanitize user-populated fields from HTML and limit content length."""
        v = bleach.clean(v, strip=True)

        # Check for empty string for user input
        role = values.get('role')
        if not v:
            raise ValueError("Message content cannot be empty.")

        # Enforce character limit of 100 if role is 'user'
        if role == 'user' and len(v) > 100:
            v = v[:100]
            logger.info(f"Truncating user input to first 100 characters. Modified input: {v}")
        return v

class AgentRequest(BaseModel):
    """Definition of the Prompt API data type."""
    messages: Optional[List[Message]] = Field([], description="A list of messages comprising the conversation so far. The roles of the messages must be alternating between user and assistant. The last input message should have role user. A message with the the system role is optional, and must be the very first message if it is present. Relevant only for api_type create_session and generate.", max_items=50000)
    user_id: Optional[str] = Field("", description="A unique identifier representing your end-user.")
    session_id: Optional[str] = Field("", description="A unique identifier representing the session associated with the response.")
    api_type: str = Field(description="The type of API action: 'create_session', 'end_session' or 'generate'.", default="create_session")
    generate_summary: bool = Field(description="Enable summary generation when api_type: end_session is invoked.", default=False)

class AgentResponseChoices(BaseModel):
    """ Definition of Chain response choices"""
    index: int = Field(default=0, ge=0, le=256, format="int64")
    message: Message = Field(default=Message())
    finish_reason: str = Field(default="", max_length=4096, pattern=r'[\s\S]*')

class AgentResponse(BaseModel):
    """Definition of Chain APIs resopnse data type"""
    id: str = Field(default="", max_length=100000, pattern=r'[\s\S]*')
    choices: List[AgentResponseChoices] = Field(default=[], max_items=256)
    session_id: str = Field(None, description="A unique identifier representing the session associated with the response.")
    sentiment: str = Field(default="", description="Any sentiment associated with this message")


@app.get("/agent/metrics", tags=["Health"])
async def get_metrics():
    return Response(content=prometheus_client.generate_latest(), media_type="text/plain")


@app.get("/agent/health", tags=["Health"], response_model=HealthResponse, responses={
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        }
    }
})
async def health_check():
    """
    Perform a Health Check

    Returns 200 when service is up. This does not check the health of downstream services.
    """

    try:
        # TODO: Check health of retrievers, analytics, datastores and NIMs as well
        target_api_url = f"{AGENT_SERVER_URL}/health"
        async with httpx.AsyncClient() as client:
            processed_response = await fetch_and_process_response(client, "GET", target_api_url)
            logger.debug(f"Response from /health endpoint of agent: {processed_response}")

        return HealthResponse(message=processed_response.get("message"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


async def fetch_and_process_response(client, method, url, params=None, json=None):
    try:
        # Perform the request to the target API
        resp = await client.request(method, url, params=params, json=json)
  
        # Check if the response was successful
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to get a response from the backend")

        # Fetch the response content (JSON) and parse it
        return resp.json()

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Timeout occurred while connecting to the backend service.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post(
    "/agent/generate",
    response_model=AgentResponse, tags=["Agent"],
    responses={
        500: {
            "description": "Internal Server Error",
            "content": {"application/json": {"example": {"detail": "Internal server error occurred"}}},
        }
    },
)
async def generate_response(request: Request, prompt: AgentRequest) -> StreamingResponse:
    """Generate and stream the response to the provided prompt."""

    api_type = prompt.api_type
    logger.info(f"\n======API Gateway called with input: {prompt.dict()}=======\n")

    def get_agent_generate_response() -> StreamingResponse:
        target_api_url = f"{AGENT_SERVER_URL}/generate/stream"

        async def response_generator():
            # Forward the request to the original API as a POST request
            async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
                async with client.stream("POST", target_api_url, json=prompt.dict()) as resp:
                    if resp.status_code != 200:
                        raise HTTPException(status_code=resp.status_code, detail="Failed to get a response from the backend")

                    # Forward the streaming response from the original API to the client
                    async for chunk in resp.aiter_text():
                        if chunk and chunk[:5] == "data:":
                            yield chunk

        # Return a streaming response to the client
        return StreamingResponse(response_generator(), media_type="text/event-stream")

    def response_generator(sentence: str, sentiment: str = ""):
        """Mock response generator to simulate streaming predefined sentence."""

        # Simulate breaking the sentence into chunks (e.g., by word)
        sentence_chunks = sentence.split()  # Split the sentence by words
        resp_id = str(uuid4())  # unique response id for every query
        # Send each chunk (word) in the response
        for chunk in sentence_chunks:
            chain_response = AgentResponse(session_id=prompt.session_id, sentiment=sentiment)
            response_choice = AgentResponseChoices(
                index=0,
                message=Message(role="assistant", content=f"{chunk} ")
            )
            chain_response.id = resp_id
            chain_response.choices.append(response_choice)
            yield "data: " + str(chain_response.json()) + "\n\n"

        # End with [DONE] response
        chain_response = AgentResponse(session_id=prompt.session_id, sentiment=sentiment)
        response_choice = AgentResponseChoices(message=Message(role="assistant", content=" "), finish_reason="[DONE]")
        chain_response.id = resp_id
        chain_response.choices.append(response_choice)
        yield "data: " + str(chain_response.json()) + "\n\n"

    try:
        if api_type == "create_session":
            logger.info("Calling /create_session API of agent MS")
            target_api_url = f"{AGENT_SERVER_URL}/create_session"
            async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
                processed_response = await fetch_and_process_response(client, "GET", target_api_url)
            logger.info(f"Response from /create_session: {processed_response}")

            prompt.session_id = processed_response.get("session_id")
            logger.info(f"Calling /generate API of agent MS with session id: {prompt.session_id}")
            return get_agent_generate_response()

        elif api_type == "end_session":
            logger.info(f"Calling /end_session API of agent MS with session id: {prompt.session_id}.")
            target_api_url = f"{AGENT_SERVER_URL}/end_session"
            params = {"session_id": prompt.session_id}
            async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
                processed_response = await fetch_and_process_response(client, "GET", target_api_url, params=params)
                logger.info(f"Response from /end_session: {processed_response}")

            # If summary is enabled, call the summary endpoint
            if prompt.generate_summary:
                logger.info(f"Calling /session/summary endpoint of analytics MS with session {prompt.session_id}")
                target_api_url = f"{ANALYTICS_SERVER_URL}/session/summary"
                async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
                    generated_summary = await fetch_and_process_response(client, "GET", target_api_url, params=params)
                    logger.info(f"Response from /session/summary: {generated_summary}")

            logger.info(f"Calling /delete_session API of agent MS with session id: {prompt.session_id}.")
            target_api_url = f"{AGENT_SERVER_URL}/delete_session"
            params = {"session_id": prompt.session_id}
            async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
                processed_response = await fetch_and_process_response(client, "GET", target_api_url, params=params)
                logger.info(f"Response from /delete_session: {processed_response}")

            if prompt.generate_summary:
                return StreamingResponse(response_generator(generated_summary.get("summary", ""), sentiment=generated_summary.get("sentiment", "")), media_type="text/event-stream")
            else:
                return StreamingResponse(response_generator(processed_response.get("message", "")), media_type="text/event-stream")

        elif api_type == "generate":
            logger.info(f"Calling /generate endpoint of agent MS with session id: {prompt.session_id}")
            return get_agent_generate_response()

        else:
            raise ValueError("Wrong api_type provided as part of the request. 'create_session', 'end_session', or 'generate'")

    except httpx.ReadTimeout as e:
        logger.error(f"HTTP Read Timeout: {e}")
        raise HTTPException(status_code=504, detail="Upstream server timeout. Please try again later.")
    except httpx.RequestError as e:
        # This will catch other request-related errors like connection issues
        logger.error(f"Request Error: {e}")
        raise HTTPException(status_code=502, detail="Error communicating with the upstream server.")
    except asyncio.CancelledError as e:
        logger.error(f"Response generation was cancelled. Details: {e}")
        raise HTTPException(status_code=500, detail=f"Server interruption before response completion: {e}")
    except Exception as e:
        logger.error(f"Internal server error. Details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
