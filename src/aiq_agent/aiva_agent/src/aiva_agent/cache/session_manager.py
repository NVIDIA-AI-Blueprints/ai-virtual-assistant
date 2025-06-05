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
Manager conversation history return relavant conversation history
based on session_id
"""

from typing import List, Optional, Dict

from aiva_agent.utils import get_config
from aiva_agent.cache.local_cache import LocalCache
from aiva_agent.cache.redis_client import RedisClient


class SessionManager:
    """
    Store the conversation between user and assistant, it's stored in format
    {"session_id": {"user_id": "", "conversation_hist": [{"role": "user/assistant", "content": "", "timestamp": ""}], "last_conversation_time: ""}}
    """

    def __init__(self, *args, **kwargs) -> None:
        db_name = get_config().cache.name
        if db_name == "redis":
            print("Using Redis client for user history")
            self.memory = RedisClient()
        elif db_name == "inmemory":
            print("Using python dict for user history")
            self.memory = LocalCache()
        else:
            raise ValueError(
                f"{db_name} in not supported. Supported type redis, inmemory")

    # TODO: Create API to get last k conversation from database instead of returning everything
    def get_conversation(self, session_id: str) -> List:
        return self.memory.get_conversation(session_id)

    def get_k_conversation(self,
                           session_id: str,
                           k_turn: Optional[int] = None) -> List:
        return self.memory.get_k_conversation(session_id, k_turn)

    def save_conversation(self, session_id: str, user_id: Optional[str],
                          conversation: List) -> bool:
        return self.memory.save_conversation(session_id, user_id, conversation)

    def is_session(self, session_id: str) -> bool:
        """Check if session_id already exist in database"""
        return self.memory.is_session(session_id)

    def get_session_info(self, session_id: str) -> Dict:
        """Retrieve complete session information from database"""
        return self.memory.get_session_info(session_id)

    def response_feedback(self, session_id: str,
                          response_feedback: float) -> bool:
        return self.memory.response_feedback(session_id, response_feedback)

    def delete_conversation(self, session_id: str):
        """Delete conversation for given session id"""
        return self.memory.delete_conversation(session_id)

    def create_session(self, session_id: str, user_id: str = ""):
        """Create a entry for given session id"""
        print("CREATING SESSION:", session_id, user_id)
        return self.memory.create_session(session_id, user_id)
