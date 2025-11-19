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
    def configure_vector_store(
        self,
    ):
        """Configure the vector store."""
        pass

    # @abstractmethod
    # async def init_vector_store(
    #     self,
    #     table_name: str = "tenant_embeddings",
    #     content_column: str = "document",
    #     metadata_json_column: str = "metadata",
    #     id_column: str = "id",
    # ):
    #     """Initialize the vector store."""
    #     pass

    @abstractmethod
    def retrieve_vector_store(
        self,
    ) -> tuple[PGVectorStore, SQLRecordManager]:
        """Retrieve the vector store."""
        pass

    @abstractmethod
    def index_documents(
        self,
        docs: list[Document],
    ) -> IndexingResult:
        """Index documents."""
        pass

    @abstractmethod
    def search_records(
        self,
        query: str,
    ) -> list[Document]:
        """Search documents."""
        pass

    @abstractmethod
    def create_index(
        self,
    ):
        pass

    @abstractmethod
    def retrieve_documents_by_file_name(self, file_name: str) -> list[str]:
        "Find files by file_name in vector store"
        pass

    @abstractmethod
    def delete_documents_by_ids(self, docs_ids: list[str]) -> list[str]:
        "Delete files by ids in vector store"
        pass

    # @abstractmethod
    # def get_documents_keys_by_source_id(self, source_id: str):
    #     """Get documents keys by source ID."""
    #     pass

    # @abstractmethod
    # def delete_documents_by_source_id(self, source_id: str):
    #     """Delete documents by source ID."""
    #     pass
