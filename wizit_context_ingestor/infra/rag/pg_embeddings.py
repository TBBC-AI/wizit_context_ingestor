from langchain_core.documents import Document
from langchain.indexes import index, SQLRecordManager
from typing import List
import logging
from langchain_postgres import PGVectorStore, PGEngine
from sqlalchemy import create_engine
from dotenv import load_dotenv
from wizit_context_ingestor.application.interfaces import EmbeddingsManager
load_dotenv()

logger = logging.getLogger(__name__)

# See docker command above to launch a postgres instance with pgvector enabled.
# connection =  os.environ.get("VECTORS_CONNECTION")
# collection_name = "documents"
# GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
# GCP_PROJECT_LOCATION = os.environ.get("GCP_PROJECT_LOCATION")
# SUPABASE_TABLE: str = os.environ.get("SUPABASE_TABLE")


class PgEmbeddingsManager(EmbeddingsManager):
    """
    Manages storage and retrieval of embeddings in PostgreSQL with pgvector extension.
    This class provides an interface to store, retrieve, and search vector embeddings
    using PostgreSQL with the pgvector extension. It uses LangChain's PGVector implementation
    to handle the underlying database operations.


    Attributes:
      embeddings_model: The embeddings model to use for generating vector embeddings
      pg_connection: The PostgreSQL connection string

    Example:
      >>> embeddings_model = VertexAIEmbeddings()
      >>> manager = PgEmbeddingsManager(
      ...     embeddings_model=embeddings_model,
      ...     pg_connection="postgresql://user:password@localhost:5432/vectordb"
      ... )
      >>> documents = [Document(page_content="Sample text", metadata={"source": "example"})]
    """

    def __init__(self, embeddings_model, pg_connection: str):
        """
          Initialize the PgEmbeddingsManager.

          Args:
              embeddings_model: The embeddings model to use for generating vector embeddings
                                (typically a LangChain embeddings model instance)
              pg_connection: The PostgreSQL connection string
                            (format: postgresql://user:password@host:port/database)

          Raises:
              Exception: If there's an error initializing the vector store
        """
        self.pg_connection = pg_connection
        self.embeddings_model = embeddings_model
        self.pg_engine = None
        self.vector_store = None
        self.record_manager = None
        try:
          self.pg_engine = PGEngine.from_connection_string(url=pg_connection)
          logger.info(f"PgEmbeddingsManager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PgEmbeddingsManager: {str(e)}")
            raise

    def configure_vector_store(
      self,
      table_name: str = "langchain_pg_embedding",
      vector_size: int = 768,
      content_column: str = "document",
      id_column: str = "id",
      metadata_json_column: str = "cmetadata",
      pg_record_manager: str = "postgres/langchain_pg_collection"
    ):
      self.pg_engine.init_vectorstore_table(
        table_name=table_name,
        vector_size=vector_size,
        content_column=content_column,
        id_column=id_column,
        metadata_json_column=metadata_json_column,
      )
      self.record_manager = SQLRecordManager(
          pg_record_manager,
          engine=create_engine(url=self.pg_connection)
      )
      # TODO move this from here
      self.record_manager.create_schema()

    def init_vector_store(
      self,
      table_name: str = "langchain_pg_embedding",
      content_column: str = "document",
      metadata_json_column: str = "cmetadata",
      id_column: str = "id",
      pg_record_manager: str = "postgres/langchain_pg_collection"
    ):
      self.vector_store = PGVectorStore.create_sync(
        embedding_service=self.embeddings_model,
        engine=self.pg_engine,
        table_name=table_name,
        content_column=content_column,
        metadata_json_column=metadata_json_column,
        id_column=id_column,
      )
      self.record_manager = SQLRecordManager(
          pg_record_manager,
          engine=create_engine(url=self.pg_connection)
      )

    def vector_store_initialized(func):
        """validate vector store initialization"""
        def wrapper(self, *args, **kwargs):
          # Common validation logic
          if self.vector_store is None:
            raise Exception("Vector store not initialized")
          if self.record_manager is None:
            raise Exception("Record manager not initialized")
          return func(self, *args, **kwargs)
        return wrapper

    @vector_store_initialized
    def index_documents(self, docs: List[Document]):
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
            logger.info(f"Indexing {len(docs)} documents in vector store")
            return index(
                docs,
                self.record_manager,
                self.vector_store,
                cleanup="incremental",
                source_id_key="source",
            )
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    @vector_store_initialized
    def get_documents_keys_by_source_id(self, source_id: str):
        """
        Get document keys by source ID from the vector store.
        """
        try:
            return self.record_manager.list_keys(group_ids=[source_id])
        except Exception as e:
            logger.error(f"Error getting documents keys by source ID: {str(e)}")
            raise

    @vector_store_initialized
    def delete_documents_by_source_id(self, source_id: str):
        """
        Delete documents by source ID from the vector store.
        """
        try:
            objects_keys = self.get_documents_keys_by_source_id(source_id)
            self.record_manager.delete_keys(objects_keys)
            self.vector_store.delete(ids=objects_keys)
        except Exception as e:
            logger.error(f"Error deleting documents by source ID: {str(e)}")
            raise

    # def get_retriever(self, search_type: str = "mmr", k: int = 20):
    #     """
    #     Get a retriever interface to the vector store for semantic search.

    #     This method returns a LangChain retriever object that can be used in retrieval
    #     pipelines, retrieval-augmented generation, and other LangChain chains.

    #     Args:
    #       search_type: The search algorithm to use. Options include:
    #                    - "similarity" (standard cosine similarity)
    #                    - "mmr" (Maximum Marginal Relevance, balances relevance with diversity)
    #                    - "similarity_score_threshold" (filters by minimum similarity)
    #       k: The number of documents to retrieve (default: 20)

    #     Returns:
    #       Retriever: A LangChain Retriever object that can be used in chains and pipelines

    #     Raises:
    #       Exception: If there's an error creating the retriever

    #     Example:
    #       >>> retriever = pg_manager.get_retriever(search_type="mmr", k=5)
    #       >>> docs = retriever.get_relevant_documents("quantum computing")
    #     """
    #     try:
    #         return self.vector_store.as_retriever(
    #             search_type=search_type, search_kwargs={"k": k}
    #         )
    #     except Exception as e:
    #         logger.info(f"failed to get vector store as retriever {str(e)}")
    #         raise
