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
based on session_id using python dict
"""

from typing import List, Optional, Dict
import time

class LocalCache:
    # Maintain conversation history session_id: List[]
    conversation_hist = {}

    # TODO: Create API to get last k conversation from database instead of returning everything
    def get_conversation(self, session_id: str) -> List:
        return self.conversation_hist.get(session_id, {}).get("conversation_hist", [])


    def get_k_conversation(self, session_id: str, k_turn: Optional[int] = None) -> List:
        if not k_turn:
            return self.conversation_hist.get(session_id, {}).get("conversation_hist", [])
        return self.conversation_hist.get(session_id, {}).get("conversation_hist", [])[-k_turn:]


    def save_conversation(self, session_id: str, user_id: Optional[str], conversation: List) -> bool:
        try:
            self.conversation_hist[session_id] = {
                "user_id": user_id or "",
                "conversation_hist": self.conversation_hist.get(session_id, {}).get("conversation_hist", []) + conversation,
                "last_conversation_time": f"{time.time()}"
            }
            return True
        except Exception as e:
            print(f"Failed to save conversation due to exception {e}")
            return False


    def is_session(self, session_id: str) -> bool:
        """Check if session_id already exist in database"""
        return session_id in self.conversation_hist


    def get_session_info(self, session_id: str) -> Dict:
        """Retrieve complete session information from database"""
        return self.conversation_hist.get(session_id, {})


    def response_feedback(self, session_id: str, response_feedback: float) -> bool:
        try:
            session = self.conversation_hist.get(session_id, {})
            conversation_hist = session.get("conversation_hist", [])

            if not conversation_hist:
                print(f"No conversation history found for session {session_id}")
                return False

            conversation_hist[-1]["feedback"] = response_feedback
            return True
        except KeyError as e:
            print(f"KeyError: Unable to store user feedback. Missing key: {e}")
            return False
        except IndexError:
            print(f"IndexError: Conversation history is empty for session {session_id}")
            return False
        except Exception as e:
            print(f"Unexpected error while storing user feedback: {e}")
            return False


    def delete_conversation(self, session_id: str) -> bool:
        """Delete conversation for given session id"""
        if session_id in self.conversation_hist:
            del self.conversation_hist[session_id]
            print(f"Deleted conversation history for session ID: {session_id}")
            return True
        else:
            print(f"No conversation history found for session ID: {session_id}")
            return False


    def create_session(self, session_id: str, user_id: str = ""):
        """Create a entry for given session id"""
        try:
            # user_id is placeholder for now
            # when create_session accept user_Id utilize this
            self.conversation_hist[session_id] = {
                "user_id": user_id or "",
                "conversation_hist": [],
                "last_conversation_time": f"{time.time()}"
            }
            return True
        except Exception as e:
            print(f"Failed to create session due to exception {e}")
            return False