from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from ..data.prompts import CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT, ContextChunk
from langchain_core.messages.human import HumanMessage
from ..workflows.context_workflow import ContextWorkflow
from typing import Dict, Any, Optional, List
from .interfaces import (
    AiApplicationService,
    PersistenceService,
    RagChunker,
    EmbeddingsManager,
)
import logging


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
        self.embeddings_manager.init_vector_store()
        self.chat_model = self.ai_application_service.load_chat_model()
        # TODO
        self.context_additional_instructions = ""
        self.metadata_source = "source"

    def _retrieve_context_chunk_in_document_with_workflow(
        self,
        workflow,
        markdown_content: str,
        chunk: Document,
        chunk_metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Retrieve context chunks in document."""
        try:
            result = workflow.invoke(
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
            # chunk.page_content = (
            #     f"Context:{result['context']}, Content:{chunk.page_content}"
            # )
            chunk.metadata["context"] = result["context"]
            if chunk_metadata:
                for key, value in chunk_metadata.items():
                    chunk.metadata[key] = value
            return chunk
        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
            raise

    # def _retrieve_context_chunk_in_document(
    #     self,
    #     markdown_content: str,
    #     chunk: Document,
    #     chunk_metadata: Optional[Dict[str, Any]] = None,
    # ) -> Document:
    #     """Retrieve context chunks in document."""
    #     try:
    #         chunk_output_parser = PydanticOutputParser(pydantic_object=ContextChunk)
    #         # Create the prompt template with image
    #         prompt = ChatPromptTemplate.from_messages(
    #             [
    #                 ("system", CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT),
    #                 (
    #                     "human",
    #                     [
    #                         {
    #                             "type": "text",
    #                             "text": f"Generate context for the following chunk: <chunk>{chunk.page_content}</chunk>,  ensure all content chunks are generated in '{self.target_language}' language",
    #                         }
    #                     ],
    #                 ),
    #             ]
    #         ).partial(
    #             document_content=markdown_content,
    #             format_instructions=chunk_output_parser.get_format_instructions(),
    #         )
    #         model_with_structured_output = self.chat_model.with_structured_output(
    #             ContextChunk
    #         )
    #         # Create the chain
    #         chain = prompt | model_with_structured_output
    #         # Process the image
    #         results = chain.invoke({})
    #         # chunk.page_content = (
    #         #     f"Context:{results.context}, Content:{chunk.page_content}"
    #         # )
    #         chunk.metadata["context"] = results.context
    #         if chunk_metadata:
    #             for key, value in chunk_metadata.items():
    #                 chunk.metadata[key] = value
    #         return chunk

    #     except Exception as e:
    #         logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
    #         raise

    # def retrieve_context_chunks_in_document(
    #     self,
    #     markdown_content: str,
    #     chunks: List[Document],
    #     chunks_metadata: Optional[Dict[str, Any]] = None,
    # ) -> List[Document]:
    #     """Retrieve context chunks in document."""
    #     try:
    #         context_chunks = list(
    #             map(
    #                 lambda chunk: self._retrieve_context_chunk_in_document(
    #                     markdown_content, chunk, chunks_metadata
    #                 ),
    #                 chunks,
    #             )
    #         )
    #         return context_chunks
    #     except Exception as e:
    #         logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
    #         raise

    def retrieve_context_chunks_in_document_with_workflow(
        self,
        markdown_content: str,
        chunks: List[Document],
        chunks_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Retrieve context chunks in document."""
        try:
            context_workflow = ContextWorkflow(
                self.chat_model, self.context_additional_instructions
            )
            compiled_context_workflow = context_workflow.gen_workflow()
            compiled_context_workflow = compiled_context_workflow.compile()
            context_chunks = list(
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
            return context_chunks
        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
            raise

    def get_context_chunks_in_document(self, file_key: str, file_tags: dict = {}):
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
            context_chunks = self.retrieve_context_chunks_in_document_with_workflow(
                markdown_content, chunks, file_tags
            )
            logger.info(f"Context chunks generated:{len(context_chunks)}")
            # upsert validation
            try:
                print(f"deleting chunks: {file_key}")
                self.delete_document_context_chunks(file_key)
            except Exception as e:
                logger.error(f"could not delete by source: {e}")
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
            self.embeddings_manager.delete_documents_by_metadata_key(
                self.metadata_source, file_key
            )
        except Exception as e:
            logger.error(f"Error delete_document_context_chunks: {str(e)}")
            raise e
