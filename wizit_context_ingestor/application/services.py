"""
Application services that coordinate between domain logic and external systems.
"""
from typing import List
from logging import getLogger
from langchain_core.documents import Document
from .interfaces import AiApplicationService, PersistenceService, RagChunker, EmbeddingsManager
from ..domain.services import ParseDocModelService
from ..domain.models import ParsedDoc
logger = getLogger(__name__)


class TranscribeDocumentService:
    """Application service for managing document transcription."""
    
    def __init__(
            self, 
            ai_application_service: AiApplicationService,
            persistence_service: PersistenceService
    ):
        self.ai_application_service = ai_application_service
        self.persistence_service = persistence_service
        self.ai_application_service.load_chat_model()


    def process_document(self, file_key: str):
        """
        Process a document by parsing it and returning the parsed content.
        """
        raw_file_path = self.persistence_service.retrieve_raw_file(file_key)
        parse_doc_model_service = ParseDocModelService(raw_file_path)
        document_pages = parse_doc_model_service.parse_document_to_base64()
        parsed_pages = []
        for page in document_pages:
            page = self.ai_application_service.parse_doc_page(page)
            parsed_pages.append(page)
        logger.info(f"Parsed {len(parsed_pages)} pages")
        parsed_document = parse_doc_model_service.create_md_content(parsed_pages)
        return parsed_pages, parsed_document
        

    def save_parsed_document(self, file_key: str, parsed_document: ParsedDoc, file_tags: dict = None):
        """
        Save the parsed document to a file.
        """
        self.persistence_service.save_parsed_document(file_key, parsed_document, file_tags)


class ContextChunksInDocumentService:
    """Application service for managing context chunks in a document."""

    def __init__(
            self, 
            ai_application_service: AiApplicationService,
            persistence_service: PersistenceService,
            rag_chunker: RagChunker,
            embeddings_manager: EmbeddingsManager
    ):
        self.ai_application_service = ai_application_service
        self.persistence_service = persistence_service
        self.rag_chunker = rag_chunker
        self.embeddings_manager = embeddings_manager
        self.embeddings_manager.init_vector_store()
        self.ai_application_service.load_chat_model()

    def get_context_chunks_in_document(self, file_key: str, file_tags: dict = None):
        """
        Get the context chunks in a document.
        """
        try:
            markdown_content = self.persistence_service.load_markdown_file_content(file_key)
            langchain_rag_document = Document(
                page_content=markdown_content, 
                metadata={
                    "source": file_key
                }
            )
            logger.info(f"Document loaded:{file_key}")
            chunks = self.rag_chunker.gen_chunks_for_document(langchain_rag_document)
            logger.info(f"Chunks generated:{len(chunks)}")
            context_chunks = self.ai_application_service.retrieve_context_chunks_in_document(markdown_content, chunks, file_tags)
            logger.info(f"Context chunks generated:{len(context_chunks)}")
            self.embeddings_manager.index_documents(context_chunks)
            return context_chunks
        except Exception as e:
            logger.error("Error get_context_chunks_in_document")
            raise e
    
    def delete_document_context_chunks(self, file_key: str):
        """
        Delete the context chunks in a document.
        """
        try:
            self.embeddings_manager.delete_documents_by_source_id(file_key)
        except Exception as e:
            logger.error(f"Error delete_document_context_chunks: {str(e)}")
            raise e
        
class ConfigureVectorStoreService:
    """Application service for configuring the vector store."""

    def __init__(self, embeddings_manager: EmbeddingsManager):
        self.embeddings_manager = embeddings_manager

    def configure_vector_store(self):
        """
        Configure the vector store.
        """
        self.embeddings_manager.configure_vector_store()