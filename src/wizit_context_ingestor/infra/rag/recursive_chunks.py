import logging
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...application.interfaces import RagChunker

logger = logging.getLogger(__name__)


class RecursiveChunks(RagChunker):
    """
    Class for chunking documents based on markdown headers.
    Uses LangChain's MarkdownHeaderTextSplitter to split documents by header structure.
    """

    __slots__ = ("embeddings_model",)

    def __init__(self, chunk_size: int = 5000, chunk_overlap: int = 500):
        """
        Initialize a recursive character-based document chunker.

        Args:
            chunk_size: Maximum size of each text chunk (default: 1000)
            chunk_overlap: Number of characters to overlap between chunks (default: 100)

        Notes:
            Splits on # (Header1), ## (Header2), and ### (Header3) with headers preserved in chunks.

        """
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
            # md_header_splits = self.markdown_splitter.split_text(document.page_content)
            chunks = self.text_splitter.split_text(document.page_content)
            filtered_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(page_content=chunk, metadata={**document.metadata})
                if document.metadata["source"]:
                    chunk_doc.id = f"{uuid.uuid4()}"
                if chunk_doc.page_content is not None and chunk_doc.page_content != "":
                    filtered_chunks.append(chunk_doc)
            logger.info(f"{len(filtered_chunks)} chunks generated successfully")
            return filtered_chunks
        except Exception as e:
            logger.error(f"Failed to get chunks: {str(e)}")
            raise
