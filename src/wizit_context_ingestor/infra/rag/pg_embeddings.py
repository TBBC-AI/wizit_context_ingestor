import asyncio
import logging

from langchain.indexes import IndexingResult, SQLRecordManager, index
from langchain_core.documents import Document
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex

# from sqlalchemy.ext.asyncio import create_async_engine
# from sqlalchemy.sql.expression import ColumnExpressionArgument
from typing_extensions import Literal

from wizit_context_ingestor.application.interfaces import EmbeddingsManager

logger = logging.getLogger(__name__)

# See docker command above to launch a postgres instance with pgvector enabled.
# connection =  os.environ.get("VECTORS_CONNECTION")
# collection_name = "documents"
# GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
# GCP_PROJECT_LOCATION = os.environ.get("GCP_PROJECT_LOCATION")
# SUPABASE_TABLE: str = os.environ.get("SUPABASE_TABLE")


class PgEngineManager:
    def __init__(
        self,
        pg_connection: str,
    ):
        self.pg_connection = pg_connection
        self.pg_engine: PGEngine

    def __enter__(self):
        try:
            self.pg_engine = PGEngine.from_connection_string(
                self.pg_connection, pool_size=1
            )
            return self
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.pg_engine:
                asyncio.run(self.pg_engine.close())
        except Exception as e:
            logger.error(f"Error closing PostgreSQL connection: {e}")


class PgVectorConnectionManager:
    def __init__(
        self,
        pg_connection: str,
        embeddings_model,
        embeddings_vectors_table_name: str,
        records_manager_table_name: str = "records_manager",
        metadata_json_column: str = "metadata",
        metadata_columns: list[str] = ["source"],
        content_column: str = "document",
        id_column: str = "id",
    ):
        self.pg_connection = pg_connection
        self.pg_engine = None
        self.vector_store: PGVectorStore | None = None
        self.metadata_json_column = metadata_json_column
        self.metadata_columns = metadata_columns
        self.content_column = content_column
        self.id_column = id_column
        self.embeddings_model = embeddings_model
        self.embeddings_vectors_table_name = embeddings_vectors_table_name
        self.records_manager_table_name = records_manager_table_name

    def __enter__(self):
        try:
            self.pg_engine = PGEngine.from_connection_string(
                self.pg_connection, pool_size=1
            )
            self.vector_store = PGVectorStore.create_sync(
                embedding_service=self.embeddings_model,
                engine=self.pg_engine,
                table_name=self.embeddings_vectors_table_name,
                content_column=self.content_column,
                metadata_json_column=self.metadata_json_column,
                metadata_columns=self.metadata_columns,
                id_column=self.id_column,
            )
            self.record_manager = SQLRecordManager(
                self.records_manager_table_name,
                db_url=self.pg_connection,
                async_mode=False,
            )
            return self
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.pg_engine:
                asyncio.run(self.pg_engine.close())
            if self.vector_store:
                self.vector_store = None
            if self.record_manager:
                self.record_manager = None
        except Exception as e:
            logger.error(f"Error closing PostgreSQL connection: {e}")


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

    def __init__(
        self,
        embeddings_model,
        pg_connection: str,
        embeddings_vectors_table_name: str,
        records_manager_table_name: str,
        vector_size: int = 768,
        content_column: str = "document",
        id_column: str = "id",
        metadata_json_column: str = "metadata",
        metadata_columns: list[str] = ["source"],
    ):
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
        # self.pg_engine = PGEngine.from_connection_string(pg_connection)
        self.embeddings_vectors_table_name = embeddings_vectors_table_name
        self.records_manager_table_name = records_manager_table_name
        self.vector_size = vector_size
        self.content_column = content_column
        self.id_column = id_column
        self.metadata_json_column = metadata_json_column
        self.metadata_columns = metadata_columns
        # self.async_engine = create_async_engine(pg_connection)
        # self.pg_engine = PGEngine.from_engine(
        #     self.async_engine
        # )
        logger.info("PgEmbeddingsManager initialized")

    def configure_vector_store(
        self,
    ):
        try:
            with PgEngineManager(self.pg_connection) as pg_engine_manager:
                pg_engine_manager.pg_engine.init_vectorstore_table(
                    table_name=self.embeddings_vectors_table_name,
                    vector_size=self.vector_size,
                    content_column=self.content_column,
                    id_column=self.id_column,
                    metadata_columns=[
                        Column(f"{metadata_column}", "VARCHAR")
                        for metadata_column in self.metadata_columns
                    ],
                    metadata_json_column=self.metadata_json_column,
                )
                record_manager = SQLRecordManager(
                    self.records_manager_table_name,
                    db_url=self.pg_connection,
                    async_mode=False,
                )
                record_manager.create_schema()
        except Exception as e:
            logger.error(f"Error configure_vector_store: {e}")
            raise

    def retrieve_vector_store(
        self,
    ) -> tuple[PGVectorStore, SQLRecordManager]:
        try:
            with PgEngineManager(self.pg_connection) as pg_engine_manager:
                self.vector_store = PGVectorStore.create_sync(
                    embedding_service=self.embeddings_model,
                    engine=pg_engine_manager.pg_engine,
                    table_name=self.embeddings_vectors_table_name,
                    content_column=self.content_column,
                    metadata_json_column=self.metadata_json_column,
                    metadata_columns=self.metadata_columns,
                    id_column=self.id_column,
                )
                self.record_manager = SQLRecordManager(
                    self.records_manager_table_name, db_url=self.pg_connection
                )
                return (self.vector_store, self.record_manager)
        except Exception as e:
            logger.error(f"Error retrieve vector store: ", e)
            raise e

    def create_user_vector_store(self):
        pass

    def create_index(self):
        if self.vector_size > 2000:
            logger.warning("Indexing for vector size > 2000 is not supported")
            raise NotImplementedError(
                "Indexing for vector size > 2000 is not supported"
            )
        try:
            with PgVectorConnectionManager(
                pg_connection=self.pg_connection,
                embeddings_vectors_table_name=self.embeddings_vectors_table_name,
                metadata_json_column=self.metadata_json_column,
                embeddings_model=self.embeddings_model,
                records_manager_table_name=self.records_manager_table_name,
            ) as connection_manager:
                index = HNSWIndex()
                if connection_manager.vector_store:
                    connection_manager.vector_store.apply_vector_index(index)
                else:
                    raise ValueError("Vector store not initialized")

        except Exception as e:
            logger.info(f"Error creating index: {e}")
            raise e

    def index_documents(
        self,
        docs: list[Document],
        cleanup: Literal["incremental", "full", "scoped_full"] | None = "incremental",
        source_id_key: str = "source",
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
            with PgVectorConnectionManager(
                pg_connection=self.pg_connection,
                embeddings_vectors_table_name=self.embeddings_vectors_table_name,
                metadata_json_column=self.metadata_json_column,
                embeddings_model=self.embeddings_model,
                records_manager_table_name=self.records_manager_table_name,
            ) as connection_manager:
                if connection_manager.vector_store:
                    return index(
                        docs,
                        connection_manager.record_manager,
                        connection_manager.vector_store,
                        cleanup=cleanup,
                        source_id_key=source_id_key,
                    )
                else:
                    raise ValueError("Vector store not initialized")
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise e

    def search_records(
        self,
        query: str,
    ) -> list[Document]:
        try:
            with PgVectorConnectionManager(
                pg_connection=self.pg_connection,
                embeddings_vectors_table_name=self.embeddings_vectors_table_name,
                metadata_json_column=self.metadata_json_column,
                embeddings_model=self.embeddings_model,
                records_manager_table_name=self.records_manager_table_name,
            ) as connection_manager:
                if connection_manager.vector_store:
                    logger.info(f"Searching for '{query}' in vector store")
                    return connection_manager.vector_store.search(
                        query=query, search_type="similarity", k=5
                    )
                else:
                    raise ValueError("Vector store not initialized")
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise e

    def retrieve_documents_by_file_name(self, file_name: str) -> list[str]:
        try:
            with PgVectorConnectionManager(
                pg_connection=self.pg_connection,
                embeddings_vectors_table_name=self.embeddings_vectors_table_name,
                metadata_json_column=self.metadata_json_column,
                embeddings_model=self.embeddings_model,
                records_manager_table_name=self.records_manager_table_name,
            ) as connection_manager:
                if connection_manager.record_manager:
                    return connection_manager.record_manager.list_keys(
                        group_ids=[file_name]
                    )
                return []
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise e

    def delete_documents_by_ids(self, docs_ids: list[str]) -> list[str]:
        try:
            with PgVectorConnectionManager(
                pg_connection=self.pg_connection,
                embeddings_vectors_table_name=self.embeddings_vectors_table_name,
                metadata_json_column=self.metadata_json_column,
                embeddings_model=self.embeddings_model,
                records_manager_table_name=self.records_manager_table_name,
            ) as connection_manager:
                if (
                    connection_manager.vector_store
                    and connection_manager.record_manager
                ):
                    connection_manager.vector_store.delete(ids=docs_ids)
                    connection_manager.record_manager.delete_keys(keys=docs_ids)
                    return docs_ids
                return []
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise e
