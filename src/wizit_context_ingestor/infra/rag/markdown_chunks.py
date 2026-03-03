import logging
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ...application.interfaces import RagChunker

logger = logging.getLogger(__name__)


class MarkdownHeadersChunks(RagChunker):
    """
    Class for chunking documents based on markdown headers.
    Uses LangChain's MarkdownHeaderTextSplitter to split documents by header structure.
    """

    __slots__ = ("embeddings_model",)

    def __init__(self, chunk_size: int = 1400, chunk_overlap: int = 100):
        """
        Initialize a markdown header-based document chunker.

        Args:
            chunk_size: Maximum size of each text chunk (default: 1000)
            chunk_overlap: Number of characters to overlap between chunks (default: 100)

        Notes:
            Splits on # (Header1), ## (Header2), and ### (Header3) with headers preserved in chunks.

        """
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header1"),
                ("##", "Header2"),
                ("###", "Header3"),
            ],
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
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
            md_header_splits = self.markdown_splitter.split_text(document.page_content)
            chunks = self.text_splitter.split_documents(md_header_splits)
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
