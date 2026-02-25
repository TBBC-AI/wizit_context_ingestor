import asyncio
import logging

from langchain_classic.indexes import SQLRecordManager
from langchain_postgres import PGEngine, PGVectorStore

logger = logging.getLogger(__name__)


class PgVectorConnectionManager:
    def __init__(
        self,
        pg_connection: str,
        embeddings_model,
        embeddings_vectors_table_name: str,
        records_manager_table_name: str = "records_manager",
        metadata_json_column: str = "metadata",
        metadata_columns: list[str] = ["source"],
        content_column: str = "document",
        id_column: str = "id",
    ):
        self.pg_connection = pg_connection
        self.pg_engine = None
        self.vector_store: PGVectorStore | None = None
        self.metadata_json_column = metadata_json_column
        self.metadata_columns = metadata_columns
        self.content_column = content_column
        self.id_column = id_column
        self.embeddings_model = embeddings_model
        self.embeddings_vectors_table_name = embeddings_vectors_table_name
        self.records_manager_table_name = records_manager_table_name

    def __enter__(self):
        try:
            self.pg_engine = PGEngine.from_connection_string(
                self.pg_connection, pool_size=1
            )
            self.vector_store = PGVectorStore.create_sync(
                embedding_service=self.embeddings_model,
                engine=self.pg_engine,
                table_name=self.embeddings_vectors_table_name,
                content_column=self.content_column,
                metadata_json_column=self.metadata_json_column,
                metadata_columns=self.metadata_columns,
                id_column=self.id_column,
            )
            self.record_manager = SQLRecordManager(
                self.records_manager_table_name,
                db_url=self.pg_connection,
                async_mode=False,
            )
            return self
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.pg_engine:
                asyncio.run(self.pg_engine.close())
            if self.vector_store:
                self.vector_store = None
            if self.record_manager:
                self.record_manager = None
        except Exception as e:
            logger.error(f"Error closing PostgreSQL connection: {e}")
