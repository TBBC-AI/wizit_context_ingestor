from langchain_core.documents import Document
from typing import Dict
# from langchain.indexes import index, SQLRecordManager
from langchain_redis import RedisConfig, RedisVectorStore
from typing import List
import logging
# from langchain_postgres import PGVectorStore, PGEngine
from sqlalchemy import create_engine
from dotenv import load_dotenv
from wizit_context_ingestor.application.interfaces import EmbeddingsManager
load_dotenv()

logger = logging.getLogger(__name__)


class RedisEmbeddingsManager(EmbeddingsManager):

    def __init__(self, embeddings_model, redis_conn_string: str, metadata_tags: dict):
        """
        Initialize the RedisEmbeddingsManager.
        Args:
            embeddings_model: The embeddings model to use for generating vector embeddings
                              (typically a LangChain embeddings model instance)
            redis_conn_string: The Redis connection string
                          (format: redis://<host>:<port>)
            metadata_tags: Tags to add as metadata to redis vector store

        Raises:
            Exception: If there's an error initializing the RedisEmbeddingsManager
        """
        self.redis_conn_string = redis_conn_string
        self.embeddings_model = embeddings_model
        self.metadata_tags_schema = []

        for tag_key in metadata_tags:
          self.metadata_tags_schema.append({
              "type": "tag",
              "name": tag_key
          })

        try:
          self.redis_config = RedisConfig(
            index_name="vector_store",
            redis_url=self.redis_conn_string,
            metadata_schema=[
              {"type": "text", "name": "context"}
            ]+self.metadata_tags_schema,
          )
          self.vector_store = RedisVectorStore(self.embeddings_model, config=self.redis_config)
          logger.info(f"RedisEmbeddingsManager initialized")
        except Exception as e:
          logger.error(f"Failed to initialize RedisEmbeddingsManager: {str(e)}")
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
        """Configure the vector store."""
        pass

    def init_vector_store(
        self,
        table_name: str = "langchain_pg_embedding",
        content_column: str = "document",
        metadata_json_column: str = "cmetadata",
        id_column: str = "id",
    ):
        """Initialize the vector store."""
        pass

    def vector_store_initialized(func):
        """validate vector store initialization"""
        def wrapper(self, *args, **kwargs):
          # Common validation logic
          if self.vector_store is None:
            raise Exception("Vector store not initialized")
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
            return self.vector_store.add_documents(docs)
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    @vector_store_initialized
    def get_documents_by_id(self, id: str):
        """
        Get document by ID from the vector store.
        """
        try:
            return self.vector_store.get_by_ids(ids=[id])
        except Exception as e:
            logger.error(f"Error getting documents by ID: {str(e)}")
            raise

    @vector_store_initialized
    def delete_documents_by_id(self, ids: list[str]):
        """
        Delete documents by ID from the vector store.
        """
        try:
            self.vector_store.delete(ids=ids)
        except Exception as e:
            logger.error(f"Error deleting documents by ID: {str(e)}")
            raise

    def get_documents_keys_by_source_id(self, source_id: str):
        """Get documents keys by source ID."""
        pass

    def delete_documents_by_source_id(self, source_id: str):
        """Delete documents by source ID."""
        pass
