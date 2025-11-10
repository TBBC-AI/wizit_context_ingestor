import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages.human import HumanMessage
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..data.prompts import CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT, ContextChunk
from ..workflows.context_workflow import ContextWorkflow
from .interfaces import (
    AiApplicationService,
    EmbeddingsManager,
    PersistenceService,
    RagChunker,
)

logger = logging.getLogger(__name__)


class ContextChunksInDocumentService:
    """
    Service for chunking documents.
    """

    def __init__(
        self,
        ai_application_service: AiApplicationService,
        persistence_service: PersistenceService,
        rag_chunker: RagChunker,
        embeddings_manager: EmbeddingsManager,
        target_language: str = "es",
    ):
        """
        Initialize the ChunkerService.
        """
        self.ai_application_service = ai_application_service
        self.persistence_service = persistence_service
        self.rag_chunker = rag_chunker
        self.embeddings_manager = embeddings_manager
        self.target_language = target_language
        # self.embeddings_manager.init_vector_store()
        self.chat_model = self.ai_application_service.load_chat_model()
        # TODO
        self.context_additional_instructions = ""
        self.metadata_source = "source"

    async def _retrieve_context_chunk_in_document_with_workflow(
        self,
        workflow,
        markdown_content: str,
        chunk: Document,
        chunk_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Retrieve context chunks in document."""
        try:
            result = await workflow.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": f"Retrieve a complete context for the following chunk: <chunk>{chunk.page_content}</chunk>,  ensure all content chunks are generated with the same document's language.",
                                },
                            ]
                        )
                    ],
                    "document_content": markdown_content,
                },
                {
                    "configurable": {
                        "transcription_accuracy_threshold": 0.95,
                        "max_transcription_retries": 2,
                    }
                },
            )
            chunk.page_content = f"<context>\n{result['context']}\n</context>\n <content>\n{chunk.page_content}\n</content>"
            # INFO: prevent context in metadata because it's already included in the chunk content, also generates issues when text is long
            # chunk.metadata["context"] = result["context"]
            if chunk_metadata is not None:
                for key, value in chunk_metadata.items():
                    chunk.metadata[key] = value
            return chunk
        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
            raise

    async def retrieve_context_chunks_in_document_with_workflow(
        self,
        markdown_content: str,
        chunks: list[Document],
        chunks_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve context chunks in document."""
        try:
            context_workflow = ContextWorkflow(
                self.chat_model, self.context_additional_instructions
            )
            compiled_context_workflow = context_workflow.gen_workflow()
            compiled_context_workflow = compiled_context_workflow.compile()
            context_chunks_workflow_invocations = list(
                map(
                    lambda chunk: self._retrieve_context_chunk_in_document_with_workflow(
                        compiled_context_workflow,
                        markdown_content,
                        chunk,
                        chunks_metadata,
                    ),
                    chunks,
                )
            )
            context_chunks = await asyncio.gather(*context_chunks_workflow_invocations)
            return context_chunks
        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
            raise

    async def get_context_chunks_in_document(
        self, file_key: str, file_tags: dict | None = None
    ):
        """
        Get the context chunks in a document.
        """
        try:
            markdown_content = self.persistence_service.load_markdown_file_content(
                file_key
            )
            langchain_rag_document = Document(
                id=file_key,
                page_content=markdown_content,
                metadata={self.metadata_source: file_key},
            )
            logger.info(f"Document loaded:{file_key}")
            chunks = self.rag_chunker.gen_chunks_for_document(langchain_rag_document)
            logger.info(f"Chunks generated:{len(chunks)}")
            context_chunks = (
                await self.retrieve_context_chunks_in_document_with_workflow(
                    markdown_content, chunks, file_tags
                )
            )
            logger.info(f"Context chunks generated:{len(context_chunks)}")
            return context_chunks
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e
