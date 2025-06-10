import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.component_ref import LLMRef

logger = logging.getLogger(__name__)


class GetProductNameConfig(FunctionBaseConfig, name="get_product_name"):
    llm_name: LLMRef = Field(..., description="LLM to use for the function")
    

@register_function(config_type=GetProductNameConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def get_purchase_history_fn(
    config: GetProductNameConfig, builder: Builder
):
    import re
    from pydantic import BaseModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage
    from aiva_agent.common.utils import get_prompts

    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm.disable_streaming = True
    llm = llm.with_config(tags=["should_stream"])

    class Product(BaseModel):
        name: str = Field(..., description="Name of the product talked about.")  

    prompts = get_prompts()

    # Define the base prompt and chain
    prompt_text = prompts.get("get_product_name")["base_prompt"]
    base_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
        ]
    )          
    
    llm = llm.with_structured_output(Product)

    base_chain = base_prompt | llm

    # Define the fallback prompt and chain
    # Check for produt name in user conversation
    fallback_prompt_text = prompts.get("get_product_name")["fallback_prompt"]
    fallback_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", fallback_prompt_text),
        ]
    )

    fallback_chain = fallback_prompt | llm    

    def _filter_products_by_name(name: str, products: list) -> list:
        # TODO: Replace this by llm call to check if that can take care of cases like
        # spelling mistakes or words which are seperated
        # TODO: Directly make sql query with wildcard
        name_lower = name.lower()

        # Check for exact match first
        exact_match = [product for product in products if product.lower() == name_lower]
        if exact_match:
            return exact_match

        # If no exact match, fall back to partial matches
        name_parts = [part for part in re.split(r'\s+', name_lower) if part.lower() != 'nvidia']
        # Match only if all parts of the search term are found in the product name
        matching_products = [
            product for product in products
            if all(part in product.lower() for part in name_parts if part)
        ]

        return matching_products

    async def _response_fn(messages: list, product_list: list) -> dict:

        # query to be used for document retrieval
        # Get the last human message instead of messages[-2]        
        last_human_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        response = await base_chain.ainvoke({"query": last_human_message})

        product_name = response.name

        # Check if product name is in query
        if product_name == 'null':
            # query to be used for document retrieval
            response = await fallback_chain.ainvoke({"messages": messages})

            product_name = response.name

        # Check if it's partial name exists or not
        if product_name == 'null':
            return {}      

        matching_products = _filter_products_by_name(product_name, product_list)

        return {
            "product_in_query": product_name,
            "products_from_purchase": list(set([product for product in matching_products]))
        }


    yield FunctionInfo.create(
        single_fn=_response_fn, 
        description=("Retrieves the recent return and order details for a user, including order ID, product name, "
                     "status, relevant dates, quantity, and amount."))

