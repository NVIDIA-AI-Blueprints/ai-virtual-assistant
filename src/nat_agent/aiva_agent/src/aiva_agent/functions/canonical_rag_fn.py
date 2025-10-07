import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.component_ref import FunctionRef

logger = logging.getLogger(__name__)


class CanonicalRAGConfig(FunctionBaseConfig, name="canonical_rag"):
    url: str = Field(default="http://localhost:8000/canonical_rag")

    

@register_function(config_type=CanonicalRAGConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def canonical_rag_fn(
    config: CanonicalRAGConfig, builder: Builder
):
    import httpx

    canonical_rag_search = f"{config.url}/search"  

    async def _response_fn(query: str, conv_history: list) -> str:
        """Use this for answering generic queries about products, specifications, warranties, usage, and issues."""

        entry_doc_search = {"query": query, "top_k": 4, "conv_history": conv_history}

        async with httpx.AsyncClient() as client:
            response = await client.post(canonical_rag_search, json=entry_doc_search)
            response.raise_for_status()
            response = response.json()

        # Extract and aggregate the content
        aggregated_content = "\n".join(chunk["content"] for chunk in response.get("chunks", []))

        return aggregated_content

    yield FunctionInfo.create(
        single_fn=_response_fn, 
        description=("Use this for answering generic queries about products, specifications, warranties, "
                     "usage, and issues."))
