import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.component_ref import FunctionRef

logger = logging.getLogger(__name__)


class HandleProductQAConfig(FunctionBaseConfig, name="handle_product_qa"):
    llm_name: str = Field(..., description="The name of the LLM to use for chat completion.")
    llm_tags: list[str] = Field(default=["should_stream"], description="Tags to apply to the LLM.")
    prompt_config_file: str = Field(default="prompt.yaml", description="The path to the prompt configuration file.")
    canonical_rag_fn: FunctionRef = Field(default="canonical_rag", 
                                          description="The url of the canonical rag service.")

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

    # Get the canonical rag function
    canonical_rag_fn = await builder.get_function(name=config.canonical_rag_fn)
    
    # Initialize the LLM
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm.disable_streaming = True
    llm = llm.with_config(tags=config.llm_tags)    

    # Get the prompts
    prompts = get_prompts(prompt_config_file=config.prompt_config_file)
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

        retrieved_content = await canonical_rag_fn.ainvoke(
            query=last_message, conv_history=previous_conversation_serialized)
        
        updated_rag_prompt = rag_prompt.format(
            chat_history=previous_conversation, context=retrieved_content)
        
        response = await llm.ainvoke(updated_rag_prompt, config)

        return {"messages": [response]}

    yield FunctionInfo.create(
        single_fn=_response_fn,
        description=("Handles product QA queries."))

