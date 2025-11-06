import logging
from typing import List

from langchain.indexes import IndexingResult, SQLRecordManager, aindex, index
from langchain_core.documents import Document
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex
from sqlalchemy.ext.asyncio import create_async_engine

from wizit_context_ingestor.application.interfaces import EmbeddingsManager

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

    __slots__ = ("embeddings_model", "pg_connection")

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
        self.vector_store = None
        self.record_manager = None
        self.async_engine = create_async_engine(pg_connection)
        self.pg_engine = PGEngine.from_engine(self.async_engine)
        logger.info("PgEmbeddingsManager initialized")

    async def configure_vector_store(
        self,
        table_name: str = "langchain_pg_embedding",
        vector_size: int = 768,
        content_column: str = "document",
        id_column: str = "id",
        metadata_json_column: str = "cmetadata",
        pg_record_manager: str = "langchain_record_manager",
    ):
        try:
            await self.pg_engine.ainit_vectorstore_table(
                table_name=table_name,
                vector_size=vector_size,
                content_column=content_column,
                id_column=id_column,
                metadata_json_column=metadata_json_column,
            )

            record_manager = SQLRecordManager(
                pg_record_manager, engine=self.async_engine, async_mode=True
            )
            await record_manager.acreate_schema()
        except Exception as e:
            logger.error(f"Error configure_vector_store: {e}")
            raise

    async def retrieve_vector_store(
        self,
        table_name: str = "langchain_pg_embedding",
        content_column: str = "document",
        metadata_json_column: str = "cmetadata",
        id_column: str = "id",
        pg_record_manager: str = "langchain_record_manager",
    ) -> tuple[PGVectorStore, SQLRecordManager]:
        try:
            vector_store = await PGVectorStore.create(
                embedding_service=self.embeddings_model,
                engine=self.pg_engine,
                table_name=table_name,
                content_column=content_column,
                metadata_json_column=metadata_json_column,
                id_column=id_column,
            )
            record_manager = SQLRecordManager(
                pg_record_manager, engine=self.async_engine, async_mode=True
            )
            await record_manager.acreate_schema()
            return (vector_store, record_manager)
        except Exception as e:
            logger.error(f"Error retrieve vector store: ", e)
            raise e

    async def retrieve_record_manager(
        self, pg_record_manager: str
    ) -> SQLRecordManager | None:
        try:
            return SQLRecordManager(
                pg_record_manager,
                engine=create_async_engine(url=self.pg_connection),
                async_mode=True,
            )
        except Exception as e:
            logger.error(f"Error retrieve record manager: ", e)
            raise e

    async def init_vector_store(
        self,
        table_name: str = "langchain_pg_embedding",
        content_column: str = "document",
        metadata_json_column: str = "cmetadata",
        id_column: str = "id",
        pg_record_manager: str = "langchain_record_manager",
    ):
        self.vector_store = await PGVectorStore.create(
            embedding_service=self.embeddings_model,
            engine=self.pg_engine,
            table_name=table_name,
            content_column=content_column,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
        )
        self.record_manager = SQLRecordManager(
            pg_record_manager,
            engine=create_async_engine(url=self.pg_connection),
            async_mode=True,
        )

    async def create_index(
        self,
        vector_store: PGVectorStore,
    ):
        try:
            index = HNSWIndex()
            await vector_store.aapply_vector_index(index)
        except Exception as e:
            logger.info(f"Error creating index: {e}")

    # def vector_store_initialized(func):
    #     """validate vector store initialization"""

    #     def validate_initialization(self, *args, **kwargs):
    #         # Common validation logic
    #         if self.vector_store is None:
    #             raise Exception("Vector store not initialized")
    #         if self.record_manager is None:
    #             raise Exception("Record manager not initialized")
    #         return func(self, *args, **kwargs)
    #     return validate_initialization
    # @vector_store_initialized
    async def index_documents(
        self,
        vector_store: PGVectorStore,
        record_manager: SQLRecordManager,
        docs: list[Document],
    ) -> IndexingResult:
        """
        Index documents in the vector store with their embeddings.

        This method takes a list of Document objects and indexes them using LangChain's
        aindex function with incremental cleanup. The documents are processed through
        the embeddings model and stored in the PostgreSQL database with pgvector.

        Args:
            vector_store: The PGVectorStore instance to use for storage
            record_manager: The SQLRecordManager instance for tracking indexed documents
            docs: A list of LangChain Document objects to index in the vector store.
                  Each Document should have page_content and metadata attributes.

        Returns:
            IndexingResult: Result object containing information about the indexing operation

        Raises:
            Exception: If there's an error during the document indexing process
        """
        try:
            logger.info(f"Indexing {len(docs)} documents in vector store")
            # await self.vector_store.aadd_documents(docs)
            return await aindex(
                docs,
                record_manager,
                vector_store,
                cleanup="incremental",
                source_id_key="source",
            )
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise e

    async def search_records(
        self,
        vector_store: PGVectorStore,
        query: str,
    ) -> list[Document]:
        try:
            logger.info(f"Searching for '{query}' in vector store")
            reply = await vector_store.asearch(
                query=query, search_type="similarity", k=1
            )
            print(reply)
            return reply
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise e

    # @vector_store_initialized
    # def get_documents_keys_by_source_id(self, source_id: str):
    #     """
    #     Get document keys by source ID from the vector store.
    #     """
    #     try:
    #         return self.record_manager.list_keys(group_ids=[source_id])
    #     except Exception as e:
    #         logger.error(f"Error getting documents keys by source ID: {str(e)}")
    #         raise

    # @vector_store_initialized
    # def delete_documents_by_source_id(self, source_id: str):
    #     """
    #     Delete documents by source ID from the vector store.
    #     """
    #     try:
    #         objects_keys = self.get_documents_keys_by_source_id(source_id)
    #         self.record_manager.delete_keys(objects_keys)
    #         self.vector_store.delete(ids=objects_keys)
    #     except Exception as e:
    #         logger.error(f"Error deleting documents by source ID: {str(e)}")
    #         raise

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
