import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class AskClarificationConfig(FunctionBaseConfig, name="ask_clarification"):
    llm_name: str = Field(..., description="The name of the LLM to use for chat completion.")
    llm_tags: list[str] = Field(default=["should_stream"], description="Tags to apply to the LLM.")
    prompt_config_file: str = Field(default="prompt.yaml", description="The path to the prompt configuration file.")

@register_function(config_type=AskClarificationConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def ask_clarification_fn(
    config: AskClarificationConfig, builder: Builder
):

    from langchain_core.messages import  ToolMessage

    from aiva_agent.utils import get_prompts
    
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm.disable_streaming = True
    # Configure the LLM
    llm = llm.with_config(tags=config.llm_tags)

    # Extract the base prompt
    prompts = get_prompts(prompt_config_file=config.prompt_config_file)
    base_prompt = prompts.get("ask_clarification")["base_prompt"]

    async def _response_fn(state: dict, config: dict) -> dict:

        previous_conversation = [m for m in state['messages'] if not isinstance(m, ToolMessage)]
        updated_base_prompt = base_prompt.format(previous_conversation=previous_conversation)

        purchase_history = state.get("user_purchase_history", [])
        if state["clarification_type"] == "no_product" and state['reason'].strip():
            followup_prompt = prompts.get("ask_clarification")["followup"]["no_product"].format(
                reason=state['reason'],
                purchase_history=purchase_history
            )
        elif not state['reason'].strip():
            followup_prompt = prompts.get("ask_clarification")["followup"]["default"].format(reason=purchase_history)
        else:
            followup_prompt = prompts.get("ask_clarification")["followup"]["default"].format(reason=state['reason'])   

        # Combine base prompt and followup prompt
        prompt = f"{updated_base_prompt} {followup_prompt}"

        response = await llm.ainvoke(prompt, config)

        return {"messages": [response]}

    yield FunctionInfo.create(
        single_fn=_response_fn,
        description=("Handles ask clarification queries."))

