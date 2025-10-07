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

"""Utility functions used across different modules of NIM Blueprints."""
import os
import yaml
import logging
from pathlib import Path
from functools import lru_cache, wraps
from urllib.parse import urlparse
from typing import TYPE_CHECKING, Callable, List, Dict

logger = logging.getLogger(__name__)

try:
    import torch
except Exception as e:
    logger.warning(f"Optional module torch not installed.")

try:
    from langchain.text_splitter import SentenceTransformersTokenTextSplitter
except Exception as e:
    logger.warning(f"Optional langchain module not installed for SentenceTransformersTokenTextSplitter.")

try:
    from langchain_core.vectorstores import VectorStore
except Exception as e:
    logger.warning(f"Optional Langchain module langchain_core not installed.")

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
except Exception as e:
    logger.error(f"Optional langchain API Catalog connector langchain_nvidia_ai_endpoints not installed.")

try:
    from langchain_community.vectorstores import PGVector
    from langchain_community.vectorstores import Milvus
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception as e:
    logger.warning(f"Optional Langchain module langchain_community not installed.")

try:
    from faiss import IndexFlatL2
except Exception as e:
    logger.warning(f"Optional faissDB not installed.")


from langchain_core.embeddings import Embeddings
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain.llms.base import LLM
from aiva_agent.common import configuration

if TYPE_CHECKING:
    from aiva_agent.common.configuration_wizard import ConfigWizard

DEFAULT_MAX_CONTEXT = 1500

def utils_cache(func: Callable) -> Callable:
    """Use this to convert unhashable args to hashable ones"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert unhashable args to hashable ones
        args_hashable = tuple(tuple(arg) if isinstance(arg, (list, dict, set)) else arg for arg in args)
        kwargs_hashable = {key: tuple(value) if isinstance(value, (list, dict, set)) else value for key, value in kwargs.items()}
        return func(*args_hashable, **kwargs_hashable)
    return wrapper


@lru_cache
def get_config_depr() -> "ConfigWizard":
    """Parse the application configuration."""
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    config = configuration.AppConfig.from_file(config_file)
    if config:
        return config
    raise RuntimeError("Unable to find configuration.")


def combine_dicts(dict_a, dict_b):
    """Combines two dictionaries recursively, prioritizing values from dict_b.

    Args:
        dict_a: The first dictionary.
        dict_b: The second dictionary.

    Returns:
        A new dictionary with combined key-value pairs.
    """

    combined_dict = dict_a.copy()  # Start with a copy of dict_a

    for key, value_b in dict_b.items():
        if key in combined_dict:
            value_a = combined_dict[key]
            # Remove the special handling for "command"
            if isinstance(value_a, dict) and isinstance(value_b, dict):
                combined_dict[key] = _combine_dicts(value_a, value_b)
            # Otherwise, replace the value from A with the value from B
            else:
                combined_dict[key] = value_b
        else:
            # Add any key not present in A
            combined_dict[key] = value_b

    return combined_dict
