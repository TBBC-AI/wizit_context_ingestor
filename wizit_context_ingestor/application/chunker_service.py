from google.oauth2 import service_account
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from ..data.prompts import CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT, ContextChunk
from typing import Dict, Any, Optional, List, Union
from .interfaces import AiApplicationService
from ..domain.models import ParsedDocPage
import logging


logger = logging.getLogger(__name__)


class ChunkerService:
    """
    Service for chunking documents.
    """

    def __init__(self, ai_application_service: AiApplicationService):
        """
        Initialize the ChunkerService.
        """
        self.ai_application_service = ai_application_service
        self.chat_model = self.ai_application_service.load_chat_model()

    def _retrieve_context_chunk_in_document(self, markdown_content: str, chunk: Document) -> ContextChunk:
        """Retrieve context chunks in document."""
        try:
            chunk_output_parser = PydanticOutputParser(pydantic_object=ContextChunk)
            # Create the prompt template with image
            prompt = ChatPromptTemplate.from_messages([
                ("system", CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT),
                (
                    "human", [{
                        "type": "text",
                            "text": f"Generate context for the following chunk: <chunk>{chunk.page_content}</chunk>"
                    }]
                ),
            ]).partial(
                document_content=markdown_content,
                format_instructions=chunk_output_parser.get_format_instructions()
            )
            model_with_structure = self.chat_model.with_structured_output(ContextChunk)
            # Create the chain
            chain = prompt | model_with_structure
            # Process the image
            results = chain.invoke({})
            print(chunk)
            chunk.page_content = f"Context:{results.context}, Content:{chunk.page_content}"
            chunk.metadata["context"] = results.context
            return chunk

        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
            raise


    def retrieve_context_chunks_in_document(self, markdown_content: str, chunks: List[Document]) -> List[Document]:
        """Retrieve context chunks in document."""
        try:
            context_chunks = list(map(
                lambda chunk: self._retrieve_context_chunk_in_document(markdown_content, chunk),
                chunks
            ))
            return context_chunks
        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
            raise

    # @contextmanager
    # def model_context(self):
    #     """
    #     Context manager for VertexModels to ensure proper resource cleanup.

    #     Example:
    #         with vertex_models.model_context():
    #             # Use vertex models here
    #     """
    #     try:
    #         yield self
    #     finally:
    #         # Clean up any resources if needed
    #         # This can be expanded based on specific cleanup requirements
    #         logger.debug("Exiting VertexModels context")
