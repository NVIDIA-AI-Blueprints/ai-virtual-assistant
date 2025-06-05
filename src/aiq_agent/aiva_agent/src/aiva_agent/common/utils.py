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
def get_config() -> "ConfigWizard":
    """Parse the application configuration."""
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    config = configuration.AppConfig.from_file(config_file)
    if config:
        return config
    raise RuntimeError("Unable to find configuration.")


@lru_cache
def get_prompts() -> Dict:
    """Retrieves prompt configurations from YAML file and return a dict.
    """

    # default config taking from prompt.yaml
    default_config_path = os.path.join("./", os.environ.get("EXAMPLE_PATH"), "prompt.yaml")
    #default_config_path = "./src/aiva_agent/prompt.yaml"
    #default_config_path = "aiva_agent/src/aiva_agent/prompt.yaml"
    default_config = {}
    if Path(default_config_path).exists():
        with open(default_config_path, 'r') as file:
            logger.info(f"Using prompts config file from: {default_config_path}")
            default_config = yaml.safe_load(file)

    config_file = os.environ.get("PROMPT_CONFIG_FILE", "/prompt.yaml")

    config = {}
    if Path(config_file).exists():
        with open(config_file, 'r') as file:
            logger.info(f"Using prompts config file from: {config_file}")
            config = yaml.safe_load(file)

    config = _combine_dicts(default_config, config)
    return config



def create_vectorstore_langchain(document_embedder, collection_name: str = "") -> VectorStore:
    """Create the vector db index for langchain."""

    config = get_config()

    if config.vector_store.name == "faiss":
        vectorstore = FAISS(document_embedder, IndexFlatL2(config.embeddings.dimensions), InMemoryDocstore(), {})
    elif config.vector_store.name == "pgvector":
        db_name = os.getenv('POSTGRES_DB', None)
        if not collection_name:
            collection_name = os.getenv('COLLECTION_NAME', "vector_db")
        logger.info(f"Using PGVector collection: {collection_name}")
        connection_string = f"postgresql://{os.getenv('POSTGRES_USER', '')}:{os.getenv('POSTGRES_PASSWORD', '')}@{config.vector_store.url}/{db_name}"
        vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=connection_string,
            embedding_function=document_embedder,
        )
    elif config.vector_store.name == "milvus":
        if not collection_name:
            collection_name = os.getenv('COLLECTION_NAME', "vector_db")
        logger.info(f"Using milvus collection: {collection_name}")
        url = urlparse(config.vector_store.url)
        vectorstore = Milvus(
            document_embedder,
            connection_args={"host": url.hostname, "port": url.port},
            collection_name=collection_name,
            index_params={"index_type": "GPU_IVF_FLAT", "metric_type": "L2", "nlist": config.vector_store.nlist},
            search_params={"nprobe": config.vector_store.nprobe},
            auto_id = True
        )
    else:
        raise ValueError(f"{config.vector_store.name} vector database is not supported")
    logger.info("Vector store created and saved.")
    return vectorstore


def get_vectorstore(vectorstore, document_embedder) -> VectorStore:
    """
    Send a vectorstore object.
    If a Vectorstore object already exists, the function returns that object.
    Otherwise, it creates a new Vectorstore object and returns it.
    """
    if vectorstore is None:
        return create_vectorstore_langchain(document_embedder)
    return vectorstore

@utils_cache
@lru_cache()
def get_llm(**kwargs) -> LLM | SimpleChatModel:
    """Create the LLM connection."""
    settings = get_config()

    logger.info(f"Using {settings.llm.model_engine} as model engine for llm. Model name: {settings.llm.model_name}")
    if settings.llm.model_engine == "nvidia-ai-endpoints":
        unused_params = [key for key in kwargs.keys() if key not in ['temperature', 'top_p', 'max_tokens']]
        if unused_params:
            logger.warning(f"The following parameters from kwargs are not supported: {unused_params} for {settings.llm.model_engine}")
        if settings.llm.server_url:
            logger.info(f"Using llm model {settings.llm.model_name} hosted at {settings.llm.server_url}")
            return ChatNVIDIA(base_url=f"http://{settings.llm.server_url}/v1",
                            model=settings.llm.model_name,
                            temperature = kwargs.get('temperature', None),
                            top_p = kwargs.get('top_p', None),
                            max_tokens = kwargs.get('max_tokens', None),
                            disable_streaming = True)
        else:
            logger.info(f"Using llm model {settings.llm.model_name} from api catalog")
            return ChatNVIDIA(model=settings.llm.model_name,
                            temperature = kwargs.get('temperature', None),
                            top_p = kwargs.get('top_p', None),
                            max_tokens = kwargs.get('max_tokens', None),
                            disable_streaming = True)
    else:
        raise RuntimeError("Unable to find any supported Large Language Model server. Supported engine name is nvidia-ai-endpoints.")


@lru_cache
def get_embedding_model() -> Embeddings:
    """Create the embedding model."""
    settings = get_config()

    logger.info(f"Using {settings.embeddings.model_engine} as model engine and {settings.embeddings.model_name} and model for embeddings")
    if settings.embeddings.model_engine == "huggingface":
        model_kwargs = {"device": "cpu"}
        if torch.cuda.is_available():
            model_kwargs["device"] = "cuda:0"

        encode_kwargs = {"normalize_embeddings": False}
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=settings.embeddings.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        # Load in a specific embedding model
        return hf_embeddings
    elif settings.embeddings.model_engine == "nvidia-ai-endpoints":
        if settings.embeddings.server_url:
            logger.info(f"Using embedding model {settings.embeddings.model_name} hosted at {settings.embeddings.server_url}")
            return NVIDIAEmbeddings(base_url=f"http://{settings.embeddings.server_url}/v1", model=settings.embeddings.model_name, truncate="END")
        else:
            logger.info(f"Using embedding model {settings.embeddings.model_name} hosted at api catalog")
            return NVIDIAEmbeddings(model=settings.embeddings.model_name, truncate="END")
    else:
        raise RuntimeError("Unable to find any supported embedding model. Supported engine is huggingface and nvidia-ai-endpoints.")

@lru_cache
def get_ranking_model() -> BaseDocumentCompressor:
    """Create the ranking model.

    Returns:
        BaseDocumentCompressor: Base class for document compressors.
    """

    settings = get_config()

    try:
        if settings.ranking.model_engine == "nvidia-ai-endpoints":
            if settings.ranking.server_url:
                logger.info(f"Using ranking model hosted at {settings.ranking.server_url}")
                return NVIDIARerank(
                    base_url=f"http://{settings.ranking.server_url}/v1", top_n=settings.retriever.top_k, truncate="END"
                )
            elif settings.ranking.model_name:
                logger.info(f"Using ranking model {settings.ranking.model_name} hosted at api catalog")
                return NVIDIARerank(model=settings.ranking.model_name, top_n=settings.retriever.top_k, truncate="END")
        else:
            logger.warning("Unable to find any supported ranking model. Supported engine is nvidia-ai-endpoints.")
    except Exception as e:
        logger.error(f"An error occurred while initializing ranking_model: {e}")
    return None


def get_text_splitter() -> SentenceTransformersTokenTextSplitter:
    """Return the token text splitter instance from langchain."""

    if get_config().text_splitter.model_name:
        embedding_model_name = get_config().text_splitter.model_name

    return SentenceTransformersTokenTextSplitter(
        model_name=embedding_model_name,
        tokens_per_chunk=get_config().text_splitter.chunk_size - 2,
        chunk_overlap=get_config().text_splitter.chunk_overlap,
    )


def get_docs_vectorstore_langchain(vectorstore: VectorStore) -> List[str]:
    """Retrieves filenames stored in the vector store implemented in LangChain."""

    settings = get_config()
    try:
        # No API availbe in LangChain for listing the docs, thus usig its private _dict
        extract_filename = lambda metadata : os.path.splitext(os.path.basename(metadata['source']))[0]
        if settings.vector_store.name == "faiss":
            in_memory_docstore = vectorstore.docstore._dict
            filenames = [extract_filename(doc.metadata) for doc in in_memory_docstore.values()]
            filenames = list(set(filenames))
            return filenames
        elif settings.vector_store.name == "pgvector":
            # No API availbe in LangChain for listing the docs, thus usig its private _make_session
            with vectorstore._make_session() as session:
                embedding_doc_store = session.query(vectorstore.EmbeddingStore.custom_id, vectorstore.EmbeddingStore.document, vectorstore.EmbeddingStore.cmetadata).all()
                filenames = set([extract_filename(metadata) for _, _, metadata in embedding_doc_store if metadata])
                return filenames
        elif settings.vector_store.name == "milvus":
            # Getting all the ID's > 0
            if vectorstore.col:
                milvus_data = vectorstore.col.query(expr="pk >= 0", output_fields=["pk","source", "text"])
                filenames = set([extract_filename(metadata) for metadata in milvus_data])
                return filenames
    except Exception as e:
        logger.error(f"Error occurred while retrieving documents: {e}")
    return []

def del_docs_vectorstore_langchain(vectorstore: VectorStore, filenames: List[str]) -> bool:
    """Delete documents from the vector index implemented in LangChain."""

    settings = get_config()
    try:
        # No other API availbe in LangChain for listing the docs, thus usig its private _dict
        extract_filename = lambda metadata : os.path.splitext(os.path.basename(metadata['source']))[0]
        if settings.vector_store.name == "faiss":
            in_memory_docstore = vectorstore.docstore._dict
            for filename in filenames:
                ids_list = [doc_id for doc_id, doc_data in in_memory_docstore.items() if extract_filename(doc_data.metadata) == filename]
                if not len(ids_list):
                    logger.info("File does not exist in the vectorstore")
                    return False
                vectorstore.delete(ids_list)
                logger.info(f"Deleted documents with filenames {filename}")
        elif settings.vector_store.name == "pgvector":
            with vectorstore._make_session() as session:
                collection = vectorstore.get_collection(session)
                filter_by = vectorstore.EmbeddingStore.collection_id == collection.uuid
                embedding_doc_store = session.query(vectorstore.EmbeddingStore.custom_id, vectorstore.EmbeddingStore.document, vectorstore.EmbeddingStore.cmetadata).filter(filter_by).all()
            for filename in filenames:
                ids_list = [doc_id for doc_id, doc_data, metadata in embedding_doc_store if extract_filename(metadata) == filename]
                if not len(ids_list):
                    logger.info("File does not exist in the vectorstore")
                    return False
                vectorstore.delete(ids_list)
                logger.info(f"Deleted documents with filenames {filename}")
        elif settings.vector_store.name == "milvus":
            # Getting all the ID's > 0
            milvus_data = vectorstore.col.query(expr="pk >= 0", output_fields=["pk","source", "text"])
            for filename in filenames:
                ids_list = [metadata["pk"] for metadata in milvus_data if extract_filename(metadata) == filename]
                if not len(ids_list):
                    logger.info("File does not exist in the vectorstore")
                    return False
                vectorstore.col.delete(f"pk in {ids_list}")
                logger.info(f"Deleted documents with filenames {filename}")
                return True
    except Exception as e:
        logger.error(f"Error occurred while deleting documents: {e}")
        return False
    return True

def _combine_dicts(dict_a, dict_b):
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
