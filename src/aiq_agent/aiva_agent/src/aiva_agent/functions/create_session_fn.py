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

logger = logging.getLogger(__name__)


class CreateSessionFnConfig(FunctionBaseConfig, name="create_session"):
    """Create a new session."""
    pass


@register_function(config_type=CreateSessionFnConfig)
async def create_session_fn(config: CreateSessionFnConfig, builder: Builder):
    from uuid import uuid4
    from pydantic import BaseModel
    from fastapi import HTTPException

    from aiva_agent.cache.session_manager import SessionManager
    from aiva_agent.datastore.datastore import Datastore

    class CreateSessionResponse(BaseModel):
        session_id: str

    session_manager = SessionManager()
    database = Datastore()

    async def _response_fn(unused: str | None = None) -> CreateSessionResponse:

        session_id = str(uuid4())

        # Ensure session_id created does not exist in cache
        if not session_manager.is_session(session_id):
            # Ensure session_id created does not exist in datastore (permanenet store like postgres)
            if not database.is_session(session_id):
                # Create a session on cache for validation
                session_manager.create_session(session_id)
                return CreateSessionResponse(session_id=session_id)

        raise HTTPException(status_code=500,
                            detail="Unable to generate session_id")

    yield FunctionInfo.create(single_fn=_response_fn)
