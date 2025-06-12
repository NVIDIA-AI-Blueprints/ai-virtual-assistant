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

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import FunctionRef

logger = logging.getLogger(__name__)


class ReturnWindowValidationFnConfig(FunctionBaseConfig, name="return_window_validation"):
    """Checks the return window for validation. Use 'YYYY-MM-DD' for the order date."""
    return_window_time: int = Field(default=15, description="The return window time in days.")

@register_function(config_type=ReturnWindowValidationFnConfig)
async def return_window_validation_fn(
    config: ReturnWindowValidationFnConfig, builder: Builder
):
    
    import os
    from datetime import datetime, timedelta

    async def _response_fn(order_date: str) -> str:

        try:
            # Parse the order date
            order_date = datetime.strptime(order_date, "%Y-%m-%d")

            # Get today's date
            today = os.environ.get('RETURN_WINDOW_CURRENT_DATE', "")

            if today:
                today = datetime.strptime(today, "%Y-%m-%d")
            else:
                today = datetime.now()

            # Parse the return window time
            return_days = int(config.return_window_time)

            # Calculate the return window end date
            return_window_end = order_date + timedelta(days=return_days)

            # Check if the product is within the return window
            if today <= return_window_end:
                days_left = (return_window_end - today).days
                return f"The product is eligible for return. {days_left} day(s) left in the return window."
            else:
                days_passed = (today - return_window_end).days
                return f"The return window has expired. It ended {days_passed} day(s) ago."
        except ValueError:
            return "Invalid date format. Please use 'YYYY-MM-DD' for the order date."        




    yield FunctionInfo.create(
        single_fn=_response_fn, 
        description="Checks the return window for validation. Use 'YYYY-MM-DD' for the order date.")

