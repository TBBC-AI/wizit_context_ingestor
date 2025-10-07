from langchain_core.documents import Document
from langchain_chroma import Chroma
import logging
from ...application.interfaces import EmbeddingsManager

# load_dotenv()

logger = logging.getLogger(__name__)


class ChromaEmbeddingsManager(EmbeddingsManager):
    __slots__ = ("embeddings_model", "collection_name")

    def __init__(
        self,
        embeddings_model,
        chroma_host=None,
        **chroma_conn_kwargs,
    ):
        """
        Initialize the ChromaEmbeddingsManager.
        Args:
            embeddings_model: The embeddings model to use for generating vector embeddings
                              (typically a LangChain embeddings model instance)
            chroma_host: The Chroma host URL

        Raises:
            Exception: If there's an error initializing the RedisEmbeddingsManager
        """
        self.embeddings_model = embeddings_model
        self.chroma_host = chroma_host
        try:
            if chroma_host:
                self.chroma = Chroma(
                    embedding_function=self.embeddings_model,
                    host=chroma_host,
                    **chroma_conn_kwargs,
                )
                logger.info("ChromaEmbeddingsManager initialized")
            else:
                self.chroma = Chroma(
                    embedding_function=self.embeddings_model, **chroma_conn_kwargs
                )
                logger.info("ChromaEmbeddingsManager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaEmbeddingsManager: {str(e)}")
            raise

    def configure_vector_store(
        self,
        table_name: str = "",
        vector_size: int = 768,
        content_column: str = "document",
        id_column: str = "id",
    ):
        """Configure the vector store."""
        pass

    def init_vector_store(
        self,
        table_name: str = "",
        content_column: str = "document",
        id_column: str = "id",
    ):
        """Initialize the vector store."""
        pass

    def index_documents(self, documents: list[Document]):
        """
        Add documents to the vector store with their embeddings.

        This method takes a list of Document objects, generates embeddings for them
        using the embeddings model, and stores both the documents and their
        embeddings in the PostgreSQL database.

        Args:
          docs: A list of LangChain Document objects to add to the vector store
                Each Document should have page_content and metadata attributes
                from langchain_core.documents import Document
        Returns:
          None

        Raises:
          Exception: If there's an error adding documents to the vector store
        """
        try:
            logger.info(f"Indexing {len(documents)} documents in vector store")
            self.chroma.add_documents(documents)
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    def get_documents_by_id(self, ids: list[str]):
        """
        Get document by ID from the vector store.
        """
        try:
            return self.chroma.get_by_ids(ids)
        except Exception as e:
            logger.error(f"Error getting documents by ID: {str(e)}")
            raise

    def delete_documents_by_id(self, ids: list[str]):
        """
        Delete documents by ID from the vector store.
        """
        try:
            self.chroma.delete(ids)
        except Exception as e:
            logger.error(f"Error deleting documents by ID: {str(e)}")
            raise

    def delete_documents_by_metadata_key(self, metadata_key: str, metadata_value: str):
        """
        Delete documents by filter from the vector store.
        """
        try:
            self.chroma.delete(where={metadata_key: metadata_value})
        except Exception as error:
            logger.error(
                f"Error deleting documents by filter: {str(filter)}, error: {error} "
            )
            raise

    def get_documents_keys_by_source_id(self, source_id: str):
        """Get documents keys by source ID."""
        pass

    def delete_documents_by_source_id(self, source_id: str):
        """Delete documents by source ID."""
        pass
