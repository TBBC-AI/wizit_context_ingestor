import logging
import uuid
from typing import Any, List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from ...application.interfaces import RagChunker

logger = logging.getLogger(__name__)


class MarkdownHeadersChunks(RagChunker):
    """
    Class for chunking documents based on markdown headers.
    Uses LangChain's MarkdownHeaderTextSplitter to split documents by header structure.
    """

    __slots__ = ("embeddings_model",)

    def __init__(self):
        """
        Initialize a markdown header-based document chunker.

        Args:
            embeddings_model: The embeddings model (currently unused but kept for interface compatibility)

        Notes:
            Splits on # (Title) and ## (Subtitle) headers with headers preserved in chunks.
        """
        self.text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Title"), ("##", "Subtitle")],
            strip_headers=False,
        )

    def gen_chunks_for_document(self, document: Document) -> List[Document]:
        """
        Split a document into chunks based on markdown header structure.

        Args:
            document: The document to split into chunks

        Returns:
            List of Document objects containing the chunked content with metadata

        Raises:
            Exception: If there's an error during the chunking process
        """
        try:
            chunks = self.text_splitter.split_text(document.page_content)
            filtered_chunks = []
            for i, chunk in enumerate(chunks):
                if document.metadata["source"]:
                    chunk.id = f"{uuid.uuid4()}"
                if chunk.page_content is not None and chunk.page_content != "":
                    filtered_chunks.append(chunk)
                chunk.metadata = {**chunk.metadata, **document.metadata}
            logger.info(f"{len(filtered_chunks)} chunks generated successfully")
            return filtered_chunks
        except Exception as e:
            logger.error(f"Failed to get chunks: {str(e)}")
            raise
