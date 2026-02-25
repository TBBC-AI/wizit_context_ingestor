import asyncio
import logging

from langchain_postgres import PGEngine

logger = logging.getLogger(__name__)


class PgEngineManager:
    def __init__(
        self,
        pg_connection: str,
    ):
        self.pg_connection = pg_connection
        self.pg_engine: PGEngine

    def __enter__(self):
        try:
            self.pg_engine = PGEngine.from_connection_string(
                self.pg_connection, pool_size=1
            )
            return self
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.pg_engine:
                asyncio.run(self.pg_engine.close())
        except Exception as e:
            logger.error(f"Error closing PostgreSQL connection: {e}")
