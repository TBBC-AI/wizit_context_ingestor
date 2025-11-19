import json
from logging import getLogger
from typing import Any, Dict, Literal

from langchain_core.documents import Document
from langsmith import Client, tracing_context

from .application.context_chunk_service import ContextChunksInDocumentService
from .application.kdb_service import KdbService
from .data.storage import StorageServices
from .infra.persistence.local_storage import LocalStorageService
from .infra.persistence.s3_storage import S3StorageService
from .infra.rag.pg_embeddings import PgEmbeddingsManager
from .infra.rag.semantic_chunks import SemanticChunks
from .infra.secrets.aws_secrets_manager import AwsSecretsManager
from .infra.vertex_model import VertexModels
from .utils.file_utils import validate_file_name_format

logger = getLogger(__name__)


# class PgKdbManager:
#     def __init__(
#         self,
#         embeddings_model,
#         kdb_params: Dict[Any, Any],
#     ):
#         self.embeddings_model = embeddings_model
#         self.kdb_params = kdb_params
#         self.pg_embeddings_manager = PgEmbeddingsManager(embeddings_model, **kdb_params)
#         self.kdb_service = KdbService(
#             self.pg_embeddings_manager,
#         )

#     def provision_vector_store(self):
#         try:
#             self.kdb_service.configure_kdb()
#             self.kdb_service.create_vector_store_hsnw_index()
#         except Exception as e:
#             logger.error(f"Error configuring vector store: {e}")


class PersistenceManager:
    def __init__(
        self,
        storage_service: Literal["s3", "local"],
        source_storage_route,
        target_storage_route,
    ):
        self.storage_service = storage_service
        self.source_storage_route = source_storage_route
        self.target_storage_route = target_storage_route

    def retrieve_storage_service(self):
        if self.storage_service == StorageServices.S3.value:
            return S3StorageService(
                origin_bucket_name=self.source_storage_route,
                target_bucket_name=self.target_storage_route,
            )
        elif self.storage_service == StorageServices.LOCAL.value:
            return LocalStorageService(
                source_storage_route=self.source_storage_route,
                target_storage_route=self.target_storage_route,
            )
        else:
            raise ValueError(f"Unsupported storage service: {self.storage_service}")


class PgKdbProvisioningManager:
    def __init__(
        self,
        gcp_project_id: str,
        gcp_project_location: str,
        gcp_secret_name: str,
        embeddings_model_id: str,
        kdb_params: Dict[Any, Any],
    ):
        self.aws_secrets_manager = AwsSecretsManager()
        vertex_gcp_sa = self.aws_secrets_manager.get_secret(gcp_secret_name)
        vertex_gcp_sa_dict = json.loads(vertex_gcp_sa)

        self.vertex_model = VertexModels(
            gcp_project_id, gcp_project_location, vertex_gcp_sa_dict
        )
        self.embeddings_model = self.vertex_model.load_embeddings_model(
            embeddings_model_id
        )

        self.pg_embeddings_manager = PgEmbeddingsManager(
            self.embeddings_model, **kdb_params
        )
        self.kdb_service = KdbService(
            self.pg_embeddings_manager,
        )

    def provision_vector_store(self):
        try:
            self.kdb_service.configure_kdb()
            self.kdb_service.create_vector_store_hsnw_index()
        except Exception as e:
            logger.error(f"Error configuring vector store: {e}")


class ChunksManager:
    def __init__(
        self,
        gcp_project_id: str,
        gcp_project_location: str,
        gcp_secret_name: str,
        langsmith_api_key: str,
        langsmith_project_name: str,
        storage_service: Literal["s3", "local"],
        kdb_service_name: Literal["pg"],
        kdb_params: Dict[Any, Any],
        llm_model_id: str = "claude-3-5-haiku@20241022",
        embeddings_model_id: str = "text-multilingual-embedding-002",
        target_language: str = "es",
    ):
        self.gcp_project_id = gcp_project_id
        self.gcp_project_location = gcp_project_location
        self.aws_secrets_manager = AwsSecretsManager()
        self.gcp_secret_name = gcp_secret_name
        self.llm_model_id = llm_model_id
        self.target_language = target_language
        self.gcp_sa_dict = self._get_gcp_sa_dict(gcp_secret_name)
        self.storage_service = storage_service
        self.kdb_params = kdb_params
        self.kdb_service_name = kdb_service_name
        self.vertex_model = self._get_vertex_model()
        self.embeddings_model = self.vertex_model.load_embeddings_model(
            embeddings_model_id
        )
        self.langsmith_api_key = langsmith_api_key
        self.langsmith_project_name = langsmith_project_name
        self.langsmith_client = Client(api_key=self.langsmith_api_key)
        self.pg_embeddings_manager = PgEmbeddingsManager(
            self.embeddings_model, **self.kdb_params
        )
        self.kdb_service = KdbService(
            self.pg_embeddings_manager,
        )
        # self.pg_kdb_manager = PgKdbManager(self.embeddings_model, self.kdb_params)
        # self.pg_embeddings_manager = self.pg_kdb_manager.pg_embeddings_manager
        # self.kdb_service = self.pg_kdb_manager.kdb_service
        self.rag_chunker = SemanticChunks(self.embeddings_model)

    def _get_gcp_sa_dict(self, gcp_secret_name: str):
        vertex_gcp_sa = self.aws_secrets_manager.get_secret(gcp_secret_name)
        vertex_gcp_sa_dict = json.loads(vertex_gcp_sa)
        return vertex_gcp_sa_dict

    def _get_vertex_model(self):
        vertex_model = VertexModels(
            self.gcp_project_id,
            self.gcp_project_location,
            self.gcp_sa_dict,
            llm_model_id=self.llm_model_id,
        )
        return vertex_model

    # def provision_vector_store(self):
    #     try:
    #         self.kdb_service.configure_kdb()
    #         self.kdb_service.create_vector_store_hsnw_index()
    #     except Exception as e:
    #         logger.error(f"Error configuring vector store: {e}")

    def index_documents_in_vector_store(self, docs: list[Document]):
        try:
            self.kdb_service.index_documents_in_vector_store(docs)
        except Exception as e:
            logger.error(f"Error indexing documents in vector store: {e}")

    def search_records(self, query: str):
        return self.kdb_service.search(query)

    def search_documents_by_file_name(self, file_name: str):
        return self.kdb_service.retrieve_documents_by_file_name(file_name)

    def delete_documents_by_file_name(self, file_name: str):
        return self.kdb_service.delete_documents_by_file_name(file_name)

    def tracing(func):
        async def gen_tracing_context(self, *args, **kwargs):
            with tracing_context(
                enabled=True,
                project_name=self.langsmith_project_name,
                client=self.langsmith_client,
            ):
                return await func(self, *args, **kwargs)

        return gen_tracing_context

    @tracing
    async def gen_context_chunks(
        self, file_key: str, source_storage_route: str, target_storage_route: str
    ):
        try:
            validate_file_name_format(file_key)
            persistence_layer = PersistenceManager(
                self.storage_service, source_storage_route, target_storage_route
            )
            persistence_service = persistence_layer.retrieve_storage_service()
            target_bucket_file_tags = {}
            if persistence_service.supports_tagging:
                target_bucket_file_tags = persistence_service.retrieve_file_tags(
                    file_key, target_storage_route
                )
            rag_chunker = SemanticChunks(self.embeddings_model)
            # kdb_manager = KdbManager(self.embeddings_model, self.kdb_params)
            # kdb_service = kdb_manager.retrieve_kdb_service()
            context_chunks_in_document_service = ContextChunksInDocumentService(
                ai_application_service=self.vertex_model,
                persistence_service=persistence_service,
                rag_chunker=rag_chunker,
                embeddings_manager=self.pg_embeddings_manager,
                target_language=self.target_language,
            )
            context_chunks = (
                await context_chunks_in_document_service.get_context_chunks_in_document(
                    file_key, target_bucket_file_tags
                )
            )
            return context_chunks
        except Exception as e:
            print(f"Error getting context chunks in document: {e}")
            raise e
