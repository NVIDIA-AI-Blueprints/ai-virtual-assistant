import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class HandleProductQAConfig(FunctionBaseConfig, name="handle_product_qa"):
    llm_name: str = Field(..., description="The name of the LLM to use for chat completion.")
    llm_tags: list[str] = Field(default=["should_stream"], description="Tags to apply to the LLM.")
    

@register_function(config_type=HandleProductQAConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def handle_product_qa_fn(
    config: HandleProductQAConfig, builder: Builder
):

    from langchain_core.runnables import RunnableConfig
    from langchain_core.prompts import MessagesPlaceholder
    from langchain_core.messages import (
        ToolMessage, HumanMessage, AIMessage, SystemMessage)
    from langchain_core.prompts.chat import ChatPromptTemplate

    from aiva_agent.utils import get_prompts
    from aiva_agent.utils import canonical_rag
    
    # Initialize the LLM
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm.disable_streaming = True
    llm = llm.with_config(tags=config.llm_tags)    

    # Get the prompts
    prompts = get_prompts()
    base_rag_prompt = prompts.get("rag_template")
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", base_rag_prompt),
            MessagesPlaceholder("chat_history") + "\n\nCONTEXT:  {context}"
        ])

    async def _response_fn(state: dict, config: dict) -> dict:

        # Extract the previous_conversation
        previous_conversation = [m for m in state['messages'] if not isinstance(m, ToolMessage) and m.content]
        message_type_map = {
            HumanMessage: "user",
            AIMessage: "assistant",
            SystemMessage: "system"
        }

        # Serialized conversation
        get_role = lambda x: message_type_map.get(type(x), None)
        previous_conversation_serialized = [{"role": get_role(m), "content": m.content} 
                                            for m in previous_conversation if m.content]
        last_message = previous_conversation_serialized[-1]['content']

        retrieved_content = canonical_rag(
            query=last_message, conv_history=previous_conversation_serialized)
        updated_rag_prompt = rag_prompt.format(
            chat_history=previous_conversation, context=retrieved_content)
        
        response = await llm.ainvoke(updated_rag_prompt, config)

        return {"messages": [response]}

    yield FunctionInfo.create(
        single_fn=_response_fn,
        description=("Handles product QA queries."))

