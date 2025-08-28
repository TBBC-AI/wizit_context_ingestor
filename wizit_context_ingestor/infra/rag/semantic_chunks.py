# check this documentation
# https://python.langchain.com/docs/how_to/semantic-chunker/
# https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
# https://python.langchain.com/docs/how_to/embed_text/
import logging
from typing import List, Any
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from wizit_context_ingestor.application.interfaces import RagChunker

logger = logging.getLogger(__name__)


class SemanticChunks(RagChunker):
    """
    Class for semantically chunking documents into smaller pieces based on semantic similarity.
    Uses LangChain's SemanticChunker to create semantically coherent document chunks.
    """

    def __init__(self, embeddings_model: Any):
        """
        Initialize a document chunker with an embeddings model.

        Args:
            embeddings_model: The embeddings model to use for semantic chunking
                             (must be compatible with LangChain's embeddings interface)

        Notes:
            By default the semantic chunker uses percentile above 95% to keep sentences separated
            and a minimum chunk size of 200 characters.
        """
        self.text_splitter = SemanticChunker(
            embeddings_model,
            buffer_size=1,
            add_start_index=True,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=200
        )

    def gen_chunks_for_document(self, document: Document) -> List[Document]:
        """
        Split a document into semantically coherent chunks.

        Args:
            documents: The text from documents to split into chunks

        Returns:
            List of strings containing the chunked content

        Raises:
            Exception: If there's an error during the chunking process
        """
        try:
            chunks = self.text_splitter.split_documents([document])
            logger.info(f"{len(chunks)} chunks generated successfully")
            return chunks
        except Exception as e:
            logger.error(f"Failed to get chunks: {str(e)}")
            raise
