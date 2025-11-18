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

"""
Baes class to store the user conversation in database permanently
"""

from typing import List, Optional
from datetime import datetime

from aiva_agent.datastore.postgres_client import PostgresClient
# from src.agent.datastore.redis_client import RedisClient


class Datastore:
    def __init__(self, db_type: str = "postgres", **kwargs):
        self._db_type = db_type
        if self._db_type == "postgres":
            print("Using postgres to store conversation history")
            self.database = PostgresClient(url=kwargs.get("url", "postgres:5432"),
                                           db_user=kwargs.get("db_user", "postgres"),
                                           db_password=kwargs.get("db_password", "postgres"),
                                           db_name=kwargs.get("db_name", "postgres"))
        # elif self._db_type == "redis":
        #     print("Using Redis to store conversation history")
        #     self.database = RedisClient()
        else:
            raise ValueError(f"{self._db_type} database in not supported. Supported type postgres")

    def store_conversation(self, session_id: str, user_id: Optional[str], conversation_history: list, last_conversation_time: str, start_conversation_time: str):
        """store conversation for given details"""
        self.database.store_conversation(session_id, user_id, conversation_history, last_conversation_time, start_conversation_time)

    def fetch_conversation(self, session_id: str):
        """fetch conversation for given session id"""
        self.database.fetch_conversation(session_id)

    def delete_conversation(self, session_id: str):
        """Delete conversation for given session id"""
        self.database.delete_conversation(session_id)

    def is_session(self, session_id: str) -> bool:
        return self.database.is_session(session_id)