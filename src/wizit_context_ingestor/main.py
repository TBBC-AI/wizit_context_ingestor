import json
from typing import Dict, Any, Literal
from .infra.vertex_model import VertexModels
from .application.transcription_service import TranscriptionService
from .application.context_chunk_service import ContextChunksInDocumentService
from .infra.persistence.s3_storage import S3StorageService
from .infra.persistence.local_storage import LocalStorageService
from .infra.rag.semantic_chunks import SemanticChunks
from .infra.rag.redis_embeddings import RedisEmbeddingsManager
from .infra.rag.chroma_embeddings import ChromaEmbeddingsManager
from .infra.secrets.aws_secrets_manager import AwsSecretsManager
from .data.storage import storage_services, StorageServices
from .data.kdb import kdb_services, KdbServices
from .utils.file_utils import has_invalid_file_name_format
from langsmith import Client, tracing_context


class KdbManager:
    def __init__(
        self, embeddings_model, kdb_service: kdb_services, kdb_params: Dict[Any, Any]
    ):
        self.kdb_service = kdb_service
        self.kdb_params = kdb_params
        self.embeddings_model = embeddings_model

    def retrieve_kdb_service(self):
        if self.kdb_service == KdbServices.REDIS.value:
            return RedisEmbeddingsManager(
                self.embeddings_model,
                **self.kdb_params,
            )
        elif self.kdb_service == KdbServices.CHROMA.value:
            return ChromaEmbeddingsManager(
                self.embeddings_model,
                **self.kdb_params,
            )
        else:
            raise ValueError(f"Unsupported kdb provider: {self.kdb_service}")


class PersistenceManager:
    def __init__(
        self,
        storage_service: storage_services,
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


class TranscriptionManager:
    def __init__(
        self,
        gcp_project_id: str,
        gcp_project_location: str,
        gcp_secret_name: str,
        langsmith_api_key: str,
        langsmith_project_name: str,
        storage_service: storage_services,
        source_storage_route: str,
        target_storage_route: str,
        llm_model_id: str = "claude-sonnet-4@20250514",
        target_language: str = "es",
        transcription_additional_instructions: str = "",
        transcription_accuracy_threshold: int = 90,
        max_transcription_retries: int = 2,
    ):
        self.gcp_project_id = gcp_project_id
        self.gcp_project_location = gcp_project_location
        self.aws_secrets_manager = AwsSecretsManager()
        self.gcp_secret_name = gcp_secret_name
        self.llm_model_id = llm_model_id
        self.target_language = target_language
        self.storage_service = storage_service
        self.source_storage_route = source_storage_route
        self.target_storage_route = target_storage_route
        self.transcription_additional_instructions = (
            transcription_additional_instructions
        )
        self.transcription_accuracy_threshold = transcription_accuracy_threshold
        self.max_transcription_retries = max_transcription_retries
        self.gcp_sa_dict = self._get_gcp_sa_dict(gcp_secret_name)
        self.vertex_model = self._get_vertex_model()
        self.langsmith_api_key = langsmith_api_key
        self.langsmith_project_name = langsmith_project_name
        self.langsmith_client = Client(api_key=self.langsmith_api_key)

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

    def tracing(func):
        def gen_tracing_context(self, *args, **kwargs):
            with tracing_context(
                enabled=True,
                project_name=self.langsmith_project_name,
                client=self.langsmith_client,
            ):
                return func(self, *args, **kwargs)

        return gen_tracing_context

    @tracing
    def transcribe_document(self, file_key: str):
        """Transcribe a document from source storage to target storage.
        This method serves as a generic interface for transcribing documents from
        various storage sources to target destinations. The specific implementation
        depends on the storage route types provided.

        Args:
            file_key (str): The unique identifier or path of the file to be transcribed.
        Returns:
            The result of the transcription process, typically the path or identifier
            of the transcribed document.

        Raises:
            Exception: If an error occurs during the transcription process.
        """
        try:
            if has_invalid_file_name_format(file_key):
                raise ValueError(
                    "Invalid file name format, do not provide special characters or spaces (instead use underscores or hyphens)"
                )
            persistence_layer = PersistenceManager(
                self.storage_service,
                self.source_storage_route,
                self.target_storage_route,
            )
            persistence_service = persistence_layer.retrieve_storage_service()

            transcribe_document_service = TranscriptionService(
                ai_application_service=self.vertex_model,
                persistence_service=persistence_service,
                target_language=self.target_language,
                transcription_additional_instructions=self.transcription_additional_instructions,
                transcription_accuracy_threshold=self.transcription_accuracy_threshold,
                max_transcription_retries=self.max_transcription_retries,
            )
            parsed_pages, parsed_document = (
                transcribe_document_service.process_document(file_key)
            )
            source_storage_file_tags = {}
            if persistence_service.supports_tagging:
                # source_storage_file_tags.tag_file(file_key, {"status": "transcribed"})
                source_storage_file_tags = persistence_service.retrieve_file_tags(
                    file_key, self.source_storage_route
                )
            transcribe_document_service.save_parsed_document(
                f"{file_key}.md", parsed_document, source_storage_file_tags
            )
            # create md document from parsed_pages
            print("parsed_pages", len(parsed_pages))
            # print("parsed_document", parsed_document)
            return f"{file_key}.md"
        except Exception as e:
            print(f"Error processing document: {e}")
            raise e


class ChunksManager:
    def __init__(
        self,
        gcp_project_id: str,
        gcp_project_location: str,
        gcp_secret_name: str,
        langsmith_api_key: str,
        langsmith_project_name: str,
        storage_service: storage_services,
        kdb_service: Literal["redis", "chroma"],
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
        self.kdb_service = kdb_service
        self.vertex_model = self._get_vertex_model()
        self.embeddings_model = self.vertex_model.load_embeddings_model(
            embeddings_model_id
        )
        self.langsmith_api_key = langsmith_api_key
        self.langsmith_project_name = langsmith_project_name
        self.langsmith_client = Client(api_key=self.langsmith_api_key)

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

    def tracing(func):
        def gen_tacing_context(self, *args, **kwargs):
            with tracing_context(
                enabled=True,
                project_name=self.langsmith_project_name,
                client=self.langsmith_client,
            ):
                return func(self, *args, **kwargs)

        return gen_tacing_context

    @tracing
    def gen_context_chunks(
        self, file_key: str, source_storage_route: str, target_storage_route: str
    ):
        try:
            if has_invalid_file_name_format(file_key):
                raise ValueError(
                    "Invalid file name format, do not provide special characters or spaces (instead use underscores or hyphens)"
                )
            persistence_layer = PersistenceManager(
                self.storage_service, source_storage_route, target_storage_route
            )
            persistence_service = persistence_layer.retrieve_storage_service()
            target_bucket_file_tags = []
            if persistence_service.supports_tagging:
                target_bucket_file_tags = persistence_service.retrieve_file_tags(
                    file_key, target_storage_route
                )
            rag_chunker = SemanticChunks(self.embeddings_model)
            kdb_manager = KdbManager(
                self.embeddings_model, self.kdb_service, self.kdb_params
            )
            kdb_service = kdb_manager.retrieve_kdb_service()
            context_chunks_in_document_service = ContextChunksInDocumentService(
                ai_application_service=self.vertex_model,
                persistence_service=persistence_service,
                rag_chunker=rag_chunker,
                embeddings_manager=kdb_service,
                target_language=self.target_language,
            )
            context_chunks = (
                context_chunks_in_document_service.get_context_chunks_in_document(
                    file_key, target_bucket_file_tags
                )
            )
            return context_chunks
        except Exception as e:
            print(f"Error getting context chunks in document: {e}")
            raise e
