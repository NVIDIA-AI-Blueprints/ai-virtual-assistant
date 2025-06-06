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
import re
import os
import logging
from typing import Dict
from pydantic import BaseModel, Field
from urllib.parse import urlparse

import requests

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
import psycopg2

from aiva_agent.common.utils import get_llm, get_prompts, get_config
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode

prompts = get_prompts()
logger = logging.getLogger(__name__)

# TODO get the default_kwargs from the Agent Server API
default_llm_kwargs = {"temperature": 0, "top_p": 0.7, "max_tokens": 1024}

canonical_rag_url = os.getenv('CANONICAL_RAG_URL', 'http://unstructured-retriever:8081')
canonical_rag_search = f"{canonical_rag_url}/search"  


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


async def get_checkpointer() -> tuple:
    settings = get_config()

    if settings.checkpointer.name == "postgres":
        print(f"Using {settings.checkpointer.name} hosted on {settings.checkpointer.url} for checkpointer")
        db_user = os.environ.get("POSTGRES_USER")
        db_password = os.environ.get("POSTGRES_PASSWORD")
        db_name = os.environ.get("POSTGRES_DB")
        db_uri = f"postgresql://{db_user}:{db_password}@{settings.checkpointer.url}/{db_name}?sslmode=disable"
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        }

        # Initialize PostgreSQL checkpointer
        pool = AsyncConnectionPool(
            conninfo=db_uri,
            min_size=2,
            kwargs=connection_kwargs,
        )
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        return checkpointer, pool
    elif settings.checkpointer.name == "inmemory":
        print(f"Using MemorySaver as checkpointer")
        return MemorySaver(), None
    else:
        raise ValueError(f"Only inmemory and postgres is supported chckpointer type")


def remove_state_from_checkpointer(session_id):

    settings = get_config()
    if settings.checkpointer.name == "postgres":
        # Handle cleanup for PostgreSQL checkpointer
        # Currently, there is no langgraph checkpointer API to remove data directly.
        # The following tables are involved in storing checkpoint data:
        # - checkpoint_blobs
        # - checkpoint_writes
        # - checkpoints
        # Note: checkpoint_migrations table can be skipped for deletion.
        try:
            app_database_url = settings.checkpointer.url

            # Parse the URL
            parsed_url = urlparse(f"//{app_database_url}", scheme='postgres')

            # Extract host and port
            host = parsed_url.hostname
            port = parsed_url.port

            # Connect to your PostgreSQL database
            connection = psycopg2.connect(
                dbname=os.getenv('POSTGRES_DB', None),
                user=os.getenv('POSTGRES_USER', None),
                password=os.getenv('POSTGRES_PASSWORD', None),
                host=host,
                port=port
            )
            cursor = connection.cursor()

            # Execute delete commands
            cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (session_id,))
            cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (session_id,))
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (session_id,))

            # Commit the changes
            connection.commit()
            logger.info(f"Deleted rows with thread_id: {session_id}")

        except Exception as e:
            logger.info(f"Error occurred while deleting data from checkpointer: {e}")
            # Optionally rollback if needed
            if connection:
                connection.rollback()
        finally:
            # Close the cursor and connection
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    else:
        # For other supported checkpointer(i.e. inmemory) we don't need cleanup
        pass

def canonical_rag(query: str, conv_history: list)  -> str:
    """Use this for answering generic queries about products, specifications, warranties, usage, and issues."""

    entry_doc_search = {"query": query, "top_k": 4, "conv_history": conv_history}
    response = requests.post(canonical_rag_search, json=entry_doc_search).json()

    # Extract and aggregate the content
    aggregated_content = "\n".join(chunk["content"] for chunk in response.get("chunks", []))

    return aggregated_content