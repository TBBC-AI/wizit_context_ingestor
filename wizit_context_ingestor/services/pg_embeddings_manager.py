from langchain_core.documents import Document
from pydantic import BaseModel
import logging
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# See docker command above to launch a postgres instance with pgvector enabled.
# connection =  os.environ.get("VECTORS_CONNECTION")
# collection_name = "documents"
# GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
# GCP_PROJECT_LOCATION = os.environ.get("GCP_PROJECT_LOCATION")
# SUPABASE_TABLE: str = os.environ.get("SUPABASE_TABLE")


class EmbeddingsModel(BaseModel):
    page_content: str
    metadata: dict
    embedding: list


class PgEmbeddingsManager:
    """
    Manages storage and retrieval of embeddings in PostgreSQL with pgvector extension.

    This class provides an interface to store, retrieve, and search vector embeddings
    using PostgreSQL with the pgvector extension. It uses LangChain's PGVector implementation
    to handle the underlying database operations.

    Attributes:
      vector_store: An instance of PGVector that manages the actual vector storage and retrieval

    Example:
      >>> embeddings_model = VertexAIEmbeddings()
      >>> manager = PgEmbeddingsManager(
      ...     embeddings_model=embeddings_model,
      ...     pg_embeddings_table="my_embeddings",
      ...     pg_connection="postgresql://user:password@localhost:5432/vectordb"
      ... )
      >>> documents = [Document(page_content="Sample text", metadata={"source": "example"})]
      >>> manager.index_documents(documents)
    """

    def __init__(self, embeddings_model, pg_embeddings_table: str, pg_connection: str):
        """
          Initialize the PgEmbeddingsManager.

          Args:
              embeddings_model: The embeddings model to use for generating vector embeddings
                                (typically a LangChain embeddings model instance)
              pg_embeddings_table: The name of the PostgreSQL table to store embeddings in
              pg_connection: The PostgreSQL connection string
                            (format: postgresql://user:password@host:port/database)

          Raises:
              Exception: If there's an error initializing the vector store
        """
        try:
            self.vector_store = PGVector(
                embeddings=embeddings_model,
                collection_name=pg_embeddings_table,
                connection=pg_connection
            )
            logger.info(f"PgEmbeddingsManager initialized with collection {pg_embeddings_table}")
        except Exception as e:
            logger.error(f"Failed to initialize PgEmbeddingsManager: {str(e)}")
            raise

    def index_documents(self, docs: Document):
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
            self.vector_store.add_documents(
                docs,
                dimensions=12
            )
        except Exception as e:
            logger.info(f"Indexed {len(docs)} documents in vector store")
            raise

    def get_retriever(self, search_type: str = "mmr", k: int = 10):
        """
        Get a retriever interface to the vector store for semantic search.

        This method returns a LangChain retriever object that can be used in retrieval
        pipelines, retrieval-augmented generation, and other LangChain chains.

        Args:
          search_type: The search algorithm to use. Options include:
                       - "similarity" (standard cosine similarity)
                       - "mmr" (Maximum Marginal Relevance, balances relevance with diversity)
                       - "similarity_score_threshold" (filters by minimum similarity)
          k: The number of documents to retrieve (default: 10)

        Returns:
          Retriever: A LangChain Retriever object that can be used in chains and pipelines

        Raises:
          Exception: If there's an error creating the retriever

        Example:
          >>> retriever = pg_manager.get_retriever(search_type="mmr", k=5)
          >>> docs = retriever.get_relevant_documents("quantum computing")
        """
        try:
            return self.vector_store.as_retriever(
                search_type=search_type, search_kwargs={"k": k}
            )
        except Exception as e:
            logger.info(f"failed to get vector store as retriever {str(e)}")
            raise