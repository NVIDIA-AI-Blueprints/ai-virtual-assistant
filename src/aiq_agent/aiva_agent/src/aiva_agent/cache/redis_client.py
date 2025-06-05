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
based on session_id using redis
"""

from typing import List, Optional, Dict
import os
import redis
import json
import time

from aiva_agent.utils import get_config

DEFAULT_DB = "0"
class RedisClient:
    def __init__(self) -> None:


        # TODO: Enable config to pass any additional config information
        # Like db etc.
        # config = eval(get_config().cache.config)
        # print("Config extracted: ", config, type(config))

        # convert hours into second as redis takes expiry time in seconds
        self.expiry = int(os.getenv("REDIS_SESSION_EXPIRY", 12)) * 60 * 60
        print(f"Redis Cache expiry {self.expiry} seconds")
        host, port = get_config().cache.url.split(":")
        # db = config.get("db", None)  or DEFAULT_DB
        print(f"Host: {host}, Port: {port}, DB: {DEFAULT_DB}")
        self.redis_client = redis.Redis(host=host, port=port, db=DEFAULT_DB, decode_responses=True)


    def get_conversation(self, session_id: str) -> List:
        """Retrieve the entire conversation history from Redis as a list"""

        conversation_hist = self.redis_client.lrange(f"{session_id}:conversation_hist", 0, -1)
        return [json.loads(conv) for conv in conversation_hist]

    def get_k_conversation(self, session_id: str, k_turn: Optional[int] = None) -> List:
        """Retrieve the last k conversations from Redis"""

        # TODO: Evaluate this implementation
        if k_turn is None:
            k_turn = -1
        conversation_hist = self.redis_client.lrange(f"{session_id}:conversation_hist", -k_turn, -1)
        return [json.loads(conv) for conv in conversation_hist]

    def save_conversation(self, session_id: str, user_id: Optional[str], conversation: List) -> bool:
        try:
            # Store each conversation entry as a JSON string in a Redis list
            for conv in conversation:
                self.redis_client.rpush(f"{session_id}:conversation_hist", json.dumps(conv))
                self.redis_client.expire(f"{session_id}:conversation_hist", self.expiry)


            # Store user_id and last conversation time as separate keys
            if user_id:
                self.redis_client.set(f"{session_id}:user_id", user_id, ex=self.expiry)

            # Store start conversation time only if it doesn't exist
            start_time_key = f"{session_id}:start_conversation_time"
            if not self.redis_client.exists(start_time_key):
                self.redis_client.set(start_time_key, f"{conversation[0].get('timestamp')}", ex=self.expiry)

            self.redis_client.set(f"{session_id}:last_conversation_time", f"{conversation[-1].get('timestamp')}", ex=self.expiry)

            start_time_key = f"{session_id}:start_conversation_time"
            if not self.redis_client.exists(start_time_key):
                self.redis_client.expire(start_time_key, self.expiry)

            return True
        except Exception as e:
            print(f"Failed to ingest document due to exception {e}")
            return False


    def is_session(self, session_id: str) -> bool:
        """Check if session_id already exist in cache"""
        return self.redis_client.exists(f"{session_id}:start_conversation_time")


    def get_session_info(self, session_id: str) -> Dict:
        """Retrieve complete session information from cache"""

        resp = {}
        conversation_hist = self.redis_client.lrange(f"{session_id}:conversation_hist", 0, -1)
        resp["conversation_hist"] = [json.loads(conv) for conv in conversation_hist]
        resp["user_id"] = self.redis_client.get(f"{session_id}:user_id")
        resp["last_conversation_time"] = self.redis_client.get(f"{session_id}:last_conversation_time")
        resp["start_conversation_time"] = self.redis_client.get(f"{session_id}:start_conversation_time")

        return resp


    def response_feedback(self, session_id: str, response_feedback: float) -> bool:
        try:
            # Get the key for the conversation history
            conv_key = f"{session_id}:conversation_hist"

            # Check if the conversation history exists
            if not self.redis_client.exists(conv_key):
                print(f"No conversation history found for session {session_id}")
                return False

            # Get the last conversation entry
            last_conv = self.redis_client.lindex(conv_key, -1)
            if not last_conv:
                print(f"Conversation history is empty for session {session_id}")
                return False

            # Parse the last conversation, add feedback, and update in Redis
            conv_data = json.loads(last_conv)
            conv_data['feedback'] = response_feedback
            updated_conv = json.dumps(conv_data)

            # Replace the last entry with the updated one
            self.redis_client.lset(conv_key, -1, updated_conv)

            return True

        except ValueError as e:
            print(f"ValueError: {str(e)}")
            return False
        except json.JSONDecodeError:
            print(f"JSONDecodeError: Unable to parse conversation data for session {session_id}")
            return False
        except redis.RedisError as e:
            print(f"RedisError: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error while storing user feedback: {str(e)}")
            return False


    def delete_conversation(self, session_id: str) -> bool:
        """Delete conversation for the given session id"""
        try:
            # Define the keys to delete
            keys_to_delete = [
                f"{session_id}:conversation_hist",
                f"{session_id}:user_id",
                f"{session_id}:last_conversation_time",
                f"{session_id}:start_conversation_time"
            ]

            # Use pipeline to delete keys, checking if they exist
            pipeline = self.redis_client.pipeline()
            for key in keys_to_delete:
                # Only delete if the key exists
                if self.redis_client.exists(key):
                    pipeline.delete(key)

            pipeline.execute()


            print(f"Deleted conversation history and associated data for session ID: {session_id}")
            return True

        except redis.RedisError as e:
            print(f"RedisError: Unable to delete conversation for session {session_id}. Error: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error while deleting conversation: {str(e)}")
            return False


    def create_session(self, session_id: str, user_id: str = ""):
        """Create a entry for given session id"""
        try:
            # Store start conversation time only if it doesn't exist
            start_time_key = f"{session_id}:start_conversation_time"
            if not self.redis_client.exists(start_time_key):
                self.redis_client.set(start_time_key, f"{time.time()}", ex=self.expiry)

            return True
        except Exception as e:
            print(f"Failed to ingest document due to exception {e}")
            return False