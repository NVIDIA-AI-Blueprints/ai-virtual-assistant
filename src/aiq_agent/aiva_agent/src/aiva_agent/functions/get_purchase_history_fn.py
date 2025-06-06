import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class GetPurchaseHistoryConfig(FunctionBaseConfig, name="get_purchase_history"):
    database_url: str = Field(default="postgres:5432", description="URL of the database")
    user: str | None = Field(default=None, description="User of the database")
    password: str = Field(default="postgres", description="Password of the database")
    dbname: str = Field(default="customer_data", description="Database name")
    

@register_function(config_type=GetPurchaseHistoryConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def get_purchase_history_fn(
    config: GetPurchaseHistoryConfig, builder: Builder
):

    from urllib.parse import urlparse
    import psycopg2
    import psycopg2.extras

    parsed_url = urlparse(f"//{config.database_url}", scheme='postgres')

    # Extract host and port
    host = parsed_url.hostname
    port = parsed_url.port

    db_params = {
        'dbname': config.dbname,
        'user': config.user,
        'password': config.password,
        'host': host,
        'port': port
    }    

    async def _response_fn(user_id: str) -> list:
        """Retrieves the recent return and order details for a user,
        including order ID, product name, status, relevant dates, quantity, and amount."""

        SQL_QUERY = f"""
        SELECT order_id, product_name, order_date, order_status, quantity, order_amount, return_status,
        return_start_date, return_received_date, return_completed_date, return_reason, notes
        FROM public.customer_data
        WHERE customer_id={user_id}
        ORDER BY order_date DESC
        LIMIT 15;
        """

        # Using context manager for connection and cursor
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(SQL_QUERY)
                result = cur.fetchall()

        # Returning result as a list of dictionaries
        return [dict(row) for row in result]

    yield FunctionInfo.create(
        single_fn=_response_fn, 
        description=("Retrieves the recent return and order details for a user, including order ID, product name, "
                     "status, relevant dates, quantity, and amount."))

