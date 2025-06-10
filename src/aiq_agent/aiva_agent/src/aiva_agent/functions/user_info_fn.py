import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.component_ref import FunctionRef

logger = logging.getLogger(__name__)


class FetchPurchaseHistoryConfig(FunctionBaseConfig, name="user_info"):
    get_purchase_history: FunctionRef = Field(default="get_purchase_history")

    

@register_function(config_type=FetchPurchaseHistoryConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def fetch_purchase_history_fn(
    config: FetchPurchaseHistoryConfig, builder: Builder
):

    get_purchase_history = builder.get_function(name=config.get_purchase_history)

    async def _response_fn(state: dict) -> dict:
        return {"user_purchase_history": await get_purchase_history.ainvoke(state["user_id"]), "current_product": ""}

    yield FunctionInfo.create(
        single_fn=_response_fn, 
        description="Fetch the purchase history for a given user."
    )
