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

logger = logging.getLogger(__name__)


class CreateSessionFnConfig(FunctionBaseConfig, name="create_session"):
    """Create a new session."""
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
    db_name: str = Field(default="postgres",
                         description="The name of the database to use. Supported types: postgres")


@register_function(config_type=CreateSessionFnConfig)
async def create_session_fn(config: CreateSessionFnConfig, builder: Builder):
    from uuid import uuid4
    from pydantic import BaseModel
    from fastapi import HTTPException

    from aiva_agent.cache.session_manager import SessionManager
    from aiva_agent.datastore.datastore import Datastore

    class CreateSessionResponse(BaseModel):
        session_id: str

    session_manager = SessionManager(cache_type=config.cache_type,
                                     host=config.cache_host,
                                     port=config.cache_port,
                                     expiry=config.cache_expiry)
    database = Datastore(db_type=config.db_type,
                         url=config.db_url,
                         db_user=config.db_user,
                         db_password=config.db_password,
                         db_name=config.db_name)

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
