import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.component_ref import FunctionRef

logger = logging.getLogger(__name__)


class ValidateProductInfoConfig(FunctionBaseConfig, name="validate_product_info"):
    get_purchase_history_fn: FunctionRef = Field(default="get_purchase_history")
    get_product_name_fn: FunctionRef = Field(default="get_product_name")


@register_function(config_type=ValidateProductInfoConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def validate_product_info_fn(
    config: ValidateProductInfoConfig, builder: Builder
):

    from langchain_core.runnables import RunnableConfig

    get_purchase_history = builder.get_function(name=config.get_purchase_history_fn) 
    get_product_name = builder.get_function(name=config.get_product_name_fn)

    async def _response_fn(state: dict, config: dict) -> dict:
        # This node will take user history and find product name based on query
        # If there are multiple name of no name specified in the graph then it will
        
        # This dict is to populate the user_purchase_history and product details if required
        response_dict = {"needs_clarification": False}
        if state["user_id"]:
            # Update user purchase history based
            response_dict.update({"user_purchase_history": await get_purchase_history.ainvoke(state["user_id"])})

            # Extracting product name which user is expecting
            product_list = list(set([resp.get("product_name") for resp in response_dict.get("user_purchase_history", [])]))

            # Extract product name from query and filter from database
            product_info = await get_product_name.ainvoke({"messages": state["messages"], "product_list": product_list})

            product_names = product_info.get("products_from_purchase", [])
            product_in_query = product_info.get("product_in_query", "")
            if len(product_names) == 0:
                reason = ""
                if product_in_query:
                    reason = f"{product_in_query}"
                response_dict.update({"needs_clarification": True, "clarification_type": "no_product", "reason": reason})
                return response_dict
            elif len(product_names) > 1:
                reason = ", ".join(product_names)
                response_dict.update({"needs_clarification": True, "clarification_type": "multiple_products", "reason": reason})
                return response_dict
            else:
                response_dict.update({"current_product": product_names[0]})

        return response_dict

    yield FunctionInfo.create(
        single_fn=_response_fn, 
        description="Search the canonical RAG for the most relevant information.")

