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
from nat.builder.context import AIQContext

logger = logging.getLogger(__name__)


class EndSessionFnConfig(FunctionBaseConfig, name="end_session"):
    """End a session"""
    cache_type: str = Field(default="redis",
                            description="The type of the cache to use. Supported types: redis, inmemory")
    db_type: str = Field(default="postgres",
                         description="The type of the database to use. Supported types: postgres")
    cache_host: str = Field(default="redis",
                            description="The host of the cache to use. Supported types: redis")
    cache_port: int = Field(default=6379,
                            description="The port of the cache to use. Supported types: redis")
    cache_expiry: int = Field(default=12,
                              description="The expiry of the cache to use. Supported types: redis")
    db_url: str = Field(default="postgres:5432",
                        description="The url of the database to use. Supported types: postgres")
    db_user: str = Field(default="postgres",
                         description="The user of the database to use. Supported types: postgres")
    db_password: str = Field(default="postgres",
                             description="The password of the database to use. Supported types: postgres")


@register_function(config_type=EndSessionFnConfig)
async def end_session_fn(
    config: EndSessionFnConfig, builder: Builder
):
    from aiva_agent.cache.session_manager import SessionManager
    from aiva_agent.datastore.datastore import Datastore
    from aiva_agent.server import EndSessionResponse
    
    session_manager = SessionManager(cache_type=config.cache_type,
                                     host=config.cache_host,
                                     port=config.cache_port,
                                     expiry=config.cache_expiry)
    database = Datastore(db_type=config.db_type,
                         url=config.db_url,
                         db_user=config.db_user,
                         db_password=config.db_password)

    async def _response_fn(message: str) -> EndSessionResponse:

        context = AIQContext.get()
        session_id = context.metadata.query_params.get("session_id")        

        logger.info(f"Fetching conversation for {session_id} from cache")
        session_info = session_manager.get_session_info(session_id)
        logger.info(f"Session INFO: {session_info}")
        if not session_info or not session_info.get("start_conversation_time", None):
            logger.info("No conversation found in session")
            return EndSessionResponse(message="Session not found. Create session before trying out")

        if session_info.get("last_conversation_time"):
            # If there is no conversation history then don't port it to datastore
            logger.info(f"Storing conversation for {session_id} in database")
            database.store_conversation(session_id, session_info.get("user_id"), session_info.get("conversation_hist"), session_info.get("last_conversation_time"), session_info.get("start_conversation_time"))

        # Once the conversation is ended and ported to permanent storage, clear cache with given session_id
        logger.info(f"Deleting conversation for {session_id} from cache")
        session_manager.delete_conversation(session_id)

        return EndSessionResponse(message="Session ended")

    yield FunctionInfo.create(single_fn=_response_fn)

