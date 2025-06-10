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
from aiq.builder.context import AIQContext

logger = logging.getLogger(__name__)


class EndSessionFnConfig(FunctionBaseConfig, name="end_session"):
    """End a session"""
    pass


@register_function(config_type=EndSessionFnConfig)
async def end_session_fn(
    config: EndSessionFnConfig, builder: Builder
):
    from aiva_agent.cache.session_manager import SessionManager
    from aiva_agent.datastore.datastore import Datastore
    from aiva_agent.server import EndSessionResponse
    
    session_manager = SessionManager()
    database = Datastore()

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

