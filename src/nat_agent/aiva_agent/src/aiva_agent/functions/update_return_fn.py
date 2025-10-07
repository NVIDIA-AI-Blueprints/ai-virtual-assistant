import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)


class UpdateReturnConfig(FunctionBaseConfig, name="update_return"):
    """Updates the return status for a given order."""
    database_url: str = Field(default="postgres:5432", description="URL of the database")
    user: str | None = Field(default=None, description="User of the database")
    password: str = Field(default="postgres", description="Password of the database")
    dbname: str = Field(default="customer_data", description="Database name")
    

@register_function(config_type=UpdateReturnConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def get_purchase_history_fn(
    config: UpdateReturnConfig, builder: Builder
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

    async def _response_fn(user_id: str, current_product: str, order_id: str) -> list:
    # Query to retrieve the order details
        SELECT_QUERY = f"""
        SELECT order_id, product_name, order_date, order_status
        FROM public.customer_data
        WHERE customer_id='{user_id}' AND product_name='{current_product}' AND order_id='{order_id}'
        ORDER BY order_date DESC
        LIMIT 1;
        """

        # Query to update the return_status
        UPDATE_QUERY = f"""
        UPDATE public.customer_data
        SET return_status = 'Requested'
        WHERE customer_id='{user_id}' AND product_name='{current_product}' AND order_id='{order_id}';
        """

        # Using context manager for connection and cursor
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Execute the SELECT query to verify the order details
                cur.execute(SELECT_QUERY)
                result = cur.fetchone()

                # If the order exists, update the return status
                if result:
                    cur.execute(UPDATE_QUERY)
                    conn.commit()  # Commit the transaction to apply the update
                    return f"Return status for order_id {order_id} has been updated to 'Requested'."
                else:
                    return f"No matching order found for user_id {user_id}, product_name {current_product}, and order_id {order_id}."        

    yield FunctionInfo.create(
        single_fn=_response_fn, 
        description=("Updates the return status for a given order."))

