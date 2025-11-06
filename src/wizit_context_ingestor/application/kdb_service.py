import logging

from langchain.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_postgres import PGVectorStore

from .interfaces import (
    EmbeddingsManager,
    RagChunker,
)

logger = logging.getLogger(__name__)


class KdbService:
    """
    Service for chunking documents.
    """

    def __init__(
        self,
        rag_chunker: RagChunker,
        embeddings_manager: EmbeddingsManager,
        embeddings_vectors_table_name: str,
        records_manager_table_name: str,
        content_column: str = "document",
        metadata_json_column: str = "metadata",
        id_column: str = "id",
        vector_size: int = 768,
    ):
        """
        Initialize the ChunkerService.
        """
        self.rag_chunker = rag_chunker
        self.embeddings_manager = embeddings_manager
        self.embeddings_vectors_table_name = embeddings_vectors_table_name
        self.records_manager_table_name = records_manager_table_name
        self.content_column = content_column
        self.vector_size = vector_size
        self.metadata_json_column = metadata_json_column
        self.id_column = id_column
        # TODO
        self.context_additional_instructions = ""
        self.metadata_source = "source"
        self._vector_store = None
        self._records_manager = None

    async def configure_kdb(self):
        await self.embeddings_manager.configure_vector_store(
            table_name=self.embeddings_vectors_table_name,
            vector_size=self.vector_size,
            content_column=self.content_column,
            id_column=self.id_column,
            metadata_json_column=self.metadata_json_column,
            pg_record_manager=self.records_manager_table_name,
        )

    async def retrieve_kdb_config(self):
        try:
            (
                vector_store_config,
                records_manager_config,
            ) = await self.embeddings_manager.retrieve_vector_store(
                table_name=self.embeddings_vectors_table_name,
                content_column=self.content_column,
                metadata_json_column=self.metadata_json_column,
                id_column=self.id_column,
                pg_record_manager=self.records_manager_table_name,
            )
            self._vector_store = vector_store_config
            self._records_manager = records_manager_config
            return vector_store_config, records_manager_config
        except Exception as e:
            logger.error(f"Error retrieving vector store: {e}")
            raise Exception(f"Error retrieving vector store: {e}")

    def check_vector_store(func):
        """validate vector store initialization"""

        async def wrapper(self, *args, **kwargs):
            # Common validation logic
            if self._vector_store is None:
                await self.retrieve_kdb_config()
            return await func(self, *args, **kwargs)

        return wrapper

    @check_vector_store
    async def create_vector_store_hsnw_index(self):
        try:
            await self.embeddings_manager.create_index(self._vector_store)
        except Exception as e:
            logger.error(f"Error creating vector store index: {e}")
            raise Exception(f"Error creating vector store index: {e}")

    @check_vector_store
    async def search(self, query: str, reintents: int = 0) -> list[Document]:
        try:
            records = []
            # if self._vector_store is None or self._records_manager is None:
            #     await self.retrieve_kdb_config()
            #     return await self.search(query, reintents=+1)
            # elif reintents > 1:
            #     raise Exception("Maximum number of retries exceeded indexing documents")
            # else:
            records = await self.embeddings_manager.search_records(
                self._vector_store, query
            )
            return records
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise Exception(f"Error indexing documents: {e}")

    @check_vector_store
    async def index_documents_in_vector_store(
        self, documents: list[Document], reintents: int = 0
    ) -> None:
        try:
            await self.embeddings_manager.index_documents(
                self._vector_store, self._records_manager, documents
            )
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise Exception(f"Error indexing documents: {e}")
