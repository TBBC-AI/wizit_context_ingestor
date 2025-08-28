"""
Application interfaces defining application layer contracts.
"""
from abc import ABC, abstractmethod
from ..domain.models import ParsedDocPage, ParsedDoc
from typing import List
from langchain_core.documents import Document


class TranscriptionService(ABC):
    """Interface for transcription services."""

    @abstractmethod
    def parse_doc_page(self, document: ParsedDocPage) -> ParsedDocPage:
        """Parse a document page."""
        pass

class AiApplicationService(ABC):    
    """Interface for AI application services."""

    @abstractmethod
    def parse_doc_page(self, document: ParsedDocPage) -> ParsedDocPage:
        """Parse a document page."""
        pass

    @abstractmethod
    def load_chat_model(self):
        """Load a chat model."""
        pass

    @abstractmethod
    def retrieve_context_chunks_in_document(self, markdown_content: str, chunks: List[Document]):
        """Retrieve context chunks in document."""
        pass


class PersistenceService(ABC):    
    """Interface for persistence services."""

    @abstractmethod
    def save_parsed_document(self, file_key: str, parsed_document: ParsedDoc):
        """Save a parsed document."""
        pass

    @abstractmethod
    def load_markdown_file_content(self, file_key: str):
        """Load markdown file content"""
        pass

    @abstractmethod
    def retrieve_raw_file(self, file_key: str):
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
        table_name: str = "langchain_pg_embedding", 
        vector_size: int = 768, 
        content_column: str = "document", 
        id_column: str = "id", 
        metadata_json_column: str = "cmetadata",
        pg_record_manager: str = "postgres/langchain_pg_collection"
    ):
        """Configure the vector store."""
        pass

    @abstractmethod
    def init_vector_store(
        self,
        table_name: str = "langchain_pg_embedding", 
        content_column: str = "document", 
        metadata_json_column: str = "cmetadata",
        id_column: str = "id",
    ):
        """Initialize the vector store."""
        pass

    @abstractmethod
    def index_documents(self, documents: List[Document]):
        """Index documents."""
        pass

    @abstractmethod
    def get_documents_keys_by_source_id(self, source_id: str):
        """Get documents keys by source ID."""
        pass

    @abstractmethod
    def delete_documents_by_source_id(self, source_id: str):
        """Delete documents by source ID."""
        pass