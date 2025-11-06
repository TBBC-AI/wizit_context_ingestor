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
        embeddings_manager: EmbeddingsManager,
    ):
        """
        Initialize the ChunkerService.
        """
        self.embeddings_manager = embeddings_manager
        self._vector_store = None
        self._records_manager = None

    def configure_kdb(self):
        try:
            self.embeddings_manager.configure_vector_store()
        except Exception as e:
            raise Exception(f"Error configuring KDB: {e}")

    def create_vector_store_hsnw_index(self):
        try:
            self.embeddings_manager.create_index()
        except Exception as e:
            logger.error(f"Error creating vector store index: {e}")
            raise Exception(f"Error creating vector store index: {e}")

    def search(self, query: str) -> list[Document]:
        try:
            records = []
            records = self.embeddings_manager.search_records(query)
            print(records)
            return records
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise Exception(f"Error indexing documents: {e}")

    def index_documents_in_vector_store(self, documents: list[Document]) -> None:
        try:
            self.embeddings_manager.index_documents(documents)
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise Exception(f"Error indexing documents: {e}")
