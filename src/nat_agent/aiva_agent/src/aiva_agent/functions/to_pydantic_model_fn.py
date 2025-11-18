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

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ToPydanticModelFnConfig(FunctionBaseConfig, name="to_pydantic_model"):
    """Convert to a ToOrderStatusAssistant pydantic model."""
    model_name: str = Field(..., description="The name of the pydantic model to convert to.")
    pass


@register_function(config_type=ToPydanticModelFnConfig)
async def to_pydantic_model_fn(
    config: ToPydanticModelFnConfig,
    builder: Builder
):

    from aiva_agent import tools

    PydanticModel = getattr(tools, config.model_name)
    
    # Get the model's fields and their types
    model_fields = PydanticModel.model_fields
    
    # Create a dynamic function with proper type hints
    def create_response_fn():
        # Get the field names and their types
        field_types = {
            name: field.annotation 
            for name, field in model_fields.items()
        }
        
        # Create the function code object
        async def _response_fn_impl(*args, **kwargs) -> PydanticModel:
            # Convert positional args to kwargs based on field order
            field_names = list(field_types.keys())
            kwargs.update(dict(zip(field_names, args)))
            return PydanticModel(**kwargs)
            
        # Set the function's annotations
        _response_fn_impl.__annotations__ = {
            **field_types,
            'return': PydanticModel
        }
        
        # Set the function's signature
        from inspect import signature, Parameter
        
        # Create parameter objects for each field
        params = [
            Parameter(
                name=name,
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=field_types[name]
            )
            for name in field_types
        ]
        
        # Create the signature
        sig = signature(_response_fn_impl).replace(parameters=params)
        _response_fn_impl.__signature__ = sig
        
        return _response_fn_impl

    _response_fn = create_response_fn()

    yield FunctionInfo.create(
        single_fn=_response_fn,
        description="Convert inputs to a pydantic model.",
    )
