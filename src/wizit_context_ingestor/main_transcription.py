import json
from logging import getLogger

from .application.transcription_app import TranscriptionApp
from .data.storage import StorageServices
from .infra.persistence.local_storage import LocalStorageService
from .infra.persistence.s3_storage import S3StorageService
from .infra.secrets.aws_secrets_manager import AwsSecretsManager
from .infra.vertex_model import VertexModels
from .utils.file_utils import validate_file_name_format

logger = getLogger(__name__)


class PersistenceManager:
    def __init__(
        self,
        storage_service: StorageServices,
        source_storage_route,
        target_storage_route,
    ):
        self.storage_service = storage_service
        self.source_storage_route = source_storage_route
        self.target_storage_route = target_storage_route

    def retrieve_storage_service(self):
        if self.storage_service == StorageServices.S3:
            return S3StorageService(
                origin_bucket_name=self.source_storage_route,
                target_bucket_name=self.target_storage_route,
            )
        elif self.storage_service == StorageServices.LOCAL:
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
        storage_service: StorageServices,
        source_storage_route: str,
        target_storage_route: str,
        llm_model_id: str = "claude-sonnet-4-6",
        target_language: str = "es",
        transcription_additional_instructions: str = "",
        transcription_accuracy_threshold: float = 0.90,
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

    async def transcribe_document(self, file_key: str):
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
            if not validate_file_name_format(file_key):
                raise ValueError(
                    "Invalid file name format, do not provide special characters or spaces (instead use underscores or hyphens)"
                )
            persistence_layer = PersistenceManager(
                self.storage_service,
                self.source_storage_route,
                self.target_storage_route,
            )
            persistence_service = persistence_layer.retrieve_storage_service()

            transcribe_document_service = TranscriptionApp(
                ai_application_service=self.vertex_model,
                persistence_service=persistence_service,
                langsmith_api_key=self.langsmith_api_key,
                langsmith_project_name=self.langsmith_project_name,
                target_language=self.target_language,
                transcription_additional_instructions=self.transcription_additional_instructions,
                transcription_accuracy_threshold=self.transcription_accuracy_threshold,
                max_transcription_retries=self.max_transcription_retries,
            )
            (
                parsed_pages,
                parsed_document,
            ) = await transcribe_document_service.process_document(file_key)
            source_storage_file_tags = {}
            if persistence_service.supports_tagging:
                # source_storage_file_tags.tag_file(file_key, {"status": "transcribed"})
                source_storage_file_tags = persistence_service.retrieve_file_tags(
                    file_key, self.source_storage_route
                )
            transcribe_document_service.save_parsed_document(
                f"{file_key}.md", parsed_document, source_storage_file_tags
            )
            return f"{file_key}.md"
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise e
