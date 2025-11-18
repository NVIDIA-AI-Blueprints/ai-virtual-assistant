import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class HandleOtherTalkConfig(FunctionBaseConfig, name="handle_other_talk"):
    llm_name: str = Field(..., description="The name of the LLM to use for chat completion.")
    llm_tags: list[str] = Field(default=["should_stream"], description="Tags to apply to the LLM.")
    prompt_config_file: str = Field(default="prompt.yaml", description="The path to the prompt configuration file.")

@register_function(config_type=HandleOtherTalkConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def handle_other_talk_fn(
    config: HandleOtherTalkConfig, builder: Builder
):

    from langchain_core.runnables import RunnableConfig
    from langchain_core.prompts.chat import ChatPromptTemplate

    from aiva_agent.utils import get_prompts
    from aiva_agent.main import State
    
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm.disable_streaming = True
    # Configure the LLM
    llm = llm.with_config(tags=config.llm_tags)    

    # Get the prompts
    prompts = get_prompts(prompt_config_file=config.prompt_config_file)
    base_prompt = prompts.get("other_talk_template", "")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", base_prompt),
            ("placeholder", "{messages}"),
        ]
    )    

    # Chain
    small_talk_chain = prompt | llm

    async def _response_fn(state: dict, config: dict) -> dict:

        # Invoke the chain
        response = await small_talk_chain.ainvoke(state, config)

        return {"messages": [response]}

    yield FunctionInfo.create(
        single_fn=_response_fn,
        description=("Handles greetings and queries outside order status, "
                     "returns, or products, providing polite redirection and "
                     "explaining chatbot limitations."))

