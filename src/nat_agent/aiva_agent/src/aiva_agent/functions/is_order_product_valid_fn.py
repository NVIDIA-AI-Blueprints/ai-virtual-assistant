import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class IsOrderProductValidConfig(FunctionBaseConfig, name="is_order_product_valid"):
    """Conditional edge from validation node to decide if we should ask followup questions"""
    pass


@register_function(config_type=IsOrderProductValidConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def is_order_product_valid_fn(
    config: IsOrderProductValidConfig, builder: Builder
):

    async def _response_fn(state: dict) -> str:
        if state["needs_clarification"] == True:
            return "ask_clarification"
        return "order_status"
    
    yield FunctionInfo.create(
        single_fn=_response_fn, 
        description="Conditional edge from validation node to decide if we should ask followup questions"
    )
