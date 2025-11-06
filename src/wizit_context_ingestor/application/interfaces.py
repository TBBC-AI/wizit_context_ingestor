"""
Application interfaces defining application layer contracts.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from langchain.indexes import IndexingResult, SQLRecordManager
from langchain_aws import ChatBedrockConverse
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_postgres import PGVectorStore

from ..domain.models import ParsedDoc, ParsedDocPage


class TranscriptionService(ABC):
    """Interface for transcription services."""

    @abstractmethod
    def parse_doc_page(self, document: ParsedDocPage) -> ParsedDocPage:
        """Parse a document page."""
        pass


class AiApplicationService(ABC):
    """Interface for AI application services."""

    # @abstractmethod
    # def parse_doc_page(self, document: ParsedDocPage) -> ParsedDocPage:
    #     """Parse a document page."""
    #     pass

    @abstractmethod
    def load_chat_model(
        self, **kwargs
    ) -> Union[ChatVertexAI, ChatAnthropicVertex, ChatBedrockConverse]:
        """Load a chat model."""
        pass

    # @abstractmethod
    # def retrieve_context_chunks_in_document(self, markdown_content: str, chunks: List[Document]):
    #     """Retrieve context chunks in document."""
    #     pass


class PersistenceService(ABC):
    """Interface for persistence services."""

    @abstractmethod
    def save_parsed_document(
        self, file_key: str, parsed_document: ParsedDoc, file_tags: Optional[dict] = {}
    ):
        """Save a parsed document."""
        pass

    @abstractmethod
    def load_markdown_file_content(self, file_key: str) -> str:
        """Load markdown file content"""
        pass

    @abstractmethod
    def retrieve_raw_file(self, file_key: str) -> str:
        """Retrieve file path in tmp folder from storage."""
        pass


class RagChunker(ABC):
    """Interface for RAG chunkers."""

    @abstractmethod
    def gen_chunks_for_document(self, document: Document) -> List[Document]:
        """Generate chunks for a document."""
        pass


class EmbeddingsManager(ABC):
    """Interface for embeddings managers."""

    @abstractmethod
    async def configure_vector_store(
        self,
        table_name: str = "tenant_embeddings",
        vector_size: int = 768,
        content_column: str = "document",
        id_column: str = "id",
        metadata_json_column: str = "cmetadata",
        pg_record_manager: str = "postgres/langchain_pg_collection",
    ):
        """Configure the vector store."""
        pass

    @abstractmethod
    async def init_vector_store(
        self,
        table_name: str = "tenant_embeddings",
        content_column: str = "document",
        metadata_json_column: str = "cmetadata",
        id_column: str = "id",
    ):
        """Initialize the vector store."""
        pass

    @abstractmethod
    async def retrieve_vector_store(
        self,
        table_name: str = "langchain_pg_embedding",
        content_column: str = "document",
        metadata_json_column: str = "cmetadata",
        id_column: str = "id",
        pg_record_manager: str = "langchain_record_manager",
    ) -> tuple[PGVectorStore, SQLRecordManager]:
        """Retrieve the vector store."""
        pass

    @abstractmethod
    async def retrieve_record_manager(
        self, pg_record_manager: str
    ) -> SQLRecordManager | None:
        pass

    @abstractmethod
    async def index_documents(
        self,
        vector_store: PGVectorStore,
        record_manager: SQLRecordManager,
        docs: list[Document],
    ) -> IndexingResult:
        """Index documents."""
        pass

    @abstractmethod
    async def search_records(
        self,
        vector_store: PGVectorStore,
        query: str,
    ) -> list[Document]:
        """Search documents."""
        pass

    @abstractmethod
    async def create_index(
        self,
        vector_store: PGVectorStore,
    ):
        pass

    # @abstractmethod
    # def get_documents_keys_by_source_id(self, source_id: str):
    #     """Get documents keys by source ID."""
    #     pass

    # @abstractmethod
    # def delete_documents_by_source_id(self, source_id: str):
    #     """Delete documents by source ID."""
    #     pass
