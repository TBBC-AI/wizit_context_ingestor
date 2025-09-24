import json
from .infra.vertex_model import VertexModels
from .application.transcription_service import TranscriptionService
from .application.context_chunk_service import ContextChunksInDocumentService
from .infra.persistence.s3_storage import S3StorageService
from .infra.persistence.local_storage import LocalStorageService
from .infra.rag.semantic_chunks import SemanticChunks
from .infra.rag.redis_embeddings import RedisEmbeddingsManager
from .infra.rag.chroma_embeddings import ChromaEmbeddingsManager
from .infra.secrets.aws_secrets_manager import AwsSecretsManager

class DeelabTranscribeManager:

    def __init__(self,
        gcp_project_id: str,
        gcp_project_location: str,
        gcp_secret_name: str,
        llm_model_id: str = "claude-sonnet-4@20250514",
        target_language: str = 'es',
        transcription_additional_instructions: str = ''
    ):
        self.gcp_project_id = gcp_project_id
        self.gcp_project_location = gcp_project_location
        self.aws_secrets_manager = AwsSecretsManager()
        self.gcp_secret_name = gcp_secret_name
        self.llm_model_id = llm_model_id
        self.target_language = target_language
        self.transcription_additional_instructions = transcription_additional_instructions
        self.gcp_sa_dict = self._get_gcp_sa_dict(gcp_secret_name)
        self.vertex_model = self._get_vertex_model()

    def _get_gcp_sa_dict(self, gcp_secret_name: str):
        vertex_gcp_sa = self.aws_secrets_manager.get_secret(gcp_secret_name)
        vertex_gcp_sa_dict = json.loads(vertex_gcp_sa)
        return vertex_gcp_sa_dict

    def _get_vertex_model(self):
        vertex_model = VertexModels(
            self.gcp_project_id,
            self.gcp_project_location,
            self.gcp_sa_dict,
            llm_model_id=self.llm_model_id
        )
        return vertex_model

    def aws_cloud_transcribe_document(
            self,
            file_key: str,
            s3_origin_bucket_name: str,
            s3_target_bucket_name: str
    ):
        try:
            s3_persistence_service = S3StorageService(
                origin_bucket_name=s3_origin_bucket_name,
                target_bucket_name=s3_target_bucket_name
            )

            transcribe_document_service = TranscriptionService(
                ai_application_service=self.vertex_model,
                persistence_service=s3_persistence_service,
                target_language=self.target_language,
                transcription_additional_instructions=self.transcription_additional_instructions
            )
            parsed_pages, parsed_document = transcribe_document_service.process_document(file_key)
            origin_bucket_file_tags = s3_persistence_service.retrieve_file_tags(file_key, s3_origin_bucket_name)
            transcribe_document_service.save_parsed_document(f"{file_key}.md", parsed_document, origin_bucket_file_tags)
            # create md document from parsed_pages
            print("parsed_pages", len(parsed_pages))
            # print("parsed_document", parsed_document)
            return f"{file_key}.md"
        except Exception as e:
            print(f"Error transcribing document: {e}")
            raise e


class DeelabRedisChunksManager:

    def __init__(
            self,
            gcp_project_id: str,
            gcp_project_location: str,
            gcp_secret_name: str,
            redis_connection_string: str,
            llm_model_id: str = "claude-3-5-haiku@20241022",
            embeddings_model_id: str = "text-multilingual-embedding-002",
            target_language: str = "es"
    ):
        self.gcp_project_id = gcp_project_id
        self.gcp_project_location = gcp_project_location
        self.aws_secrets_manager = AwsSecretsManager()
        self.gcp_secret_name = gcp_secret_name
        self.llm_model_id = llm_model_id
        self.target_language = target_language
        self.gcp_sa_dict = self._get_gcp_sa_dict(gcp_secret_name)
        self.redis_connection_string = redis_connection_string
        self.vertex_model = self._get_vertex_model()
        self.embeddings_model = self.vertex_model.load_embeddings_model(embeddings_model_id)

    def _get_gcp_sa_dict(self, gcp_secret_name: str):
        vertex_gcp_sa = self.aws_secrets_manager.get_secret(gcp_secret_name)
        vertex_gcp_sa_dict = json.loads(vertex_gcp_sa)
        return vertex_gcp_sa_dict

    def _get_vertex_model(self):
        vertex_model = VertexModels(
            self.gcp_project_id,
            self.gcp_project_location,
            self.gcp_sa_dict,
            llm_model_id=self.llm_model_id
        )
        return vertex_model

    def context_chunks_in_document(
        self,
        file_key: str
    ):
        try:
            rag_chunker = SemanticChunks(self.embeddings_model)
            redis_embeddings_manager = RedisEmbeddingsManager(
                self.embeddings_model,
                self.redis_connection_string,
                {
                    "file_key": file_key
                }
            )
            local_persistence_service = LocalStorageService()
            context_chunks_in_document_service = ContextChunksInDocumentService(
                ai_application_service=self.vertex_model,
                persistence_service=local_persistence_service,
                rag_chunker=rag_chunker,
                embeddings_manager=redis_embeddings_manager,
                target_language=self.target_language
            )
            context_chunks = context_chunks_in_document_service.get_context_chunks_in_document(file_key)
            print("context_chunks", context_chunks)
            return context_chunks
        except Exception as e:
            print(f"Error getting context chunks in document: {e}")
            raise e

    # TODO
    def context_chunks_in_document_from_aws_cloud(
            self,
            file_key: str,
            s3_origin_bucket_name: str,
            s3_target_bucket_name: str
        ):
        try:
            s3_persistence_service = S3StorageService(
                origin_bucket_name=s3_origin_bucket_name,
                target_bucket_name=s3_target_bucket_name
            )
            target_bucket_file_tags = s3_persistence_service.retrieve_file_tags(file_key, s3_target_bucket_name)

            rag_chunker = SemanticChunks(self.embeddings_model)
            redis_embeddings_manager = RedisEmbeddingsManager(
                embeddings_model=self.embeddings_model,
                redis_conn_string=self.redis_connection_string,
                metadata_tags=target_bucket_file_tags
            )
            context_chunks_in_document_service = ContextChunksInDocumentService(
                ai_application_service=self.vertex_model,
                persistence_service=s3_persistence_service,
                rag_chunker=rag_chunker,
                embeddings_manager=redis_embeddings_manager,
                target_language=self.target_language
            )
            context_chunks = context_chunks_in_document_service.get_context_chunks_in_document(file_key, target_bucket_file_tags)
            return context_chunks
        except Exception as e:
            print(f"Error getting context chunks in document: {e}")
            raise e


    def delete_document_context_chunks_from_aws_cloud(
            self,
            file_key: str,
            s3_origin_bucket_name: str,
            s3_target_bucket_name: str
        ):
        pass
        # rag_chunker = SemanticChunks(self.embeddings_model)
        # pg_embeddings_manager = PgEmbeddingsManager(
        #     embeddings_model=self.embeddings_model,
        #     pg_connection=self.vector_store_connection
        # )
        # s3_persistence_service = S3StorageService(
        #     origin_bucket_name=s3_origin_bucket_name,
        #     target_bucket_name=s3_target_bucket_name
        # )
        # context_chunks_in_document_service = ContextChunksInDocumentService(
        #     ai_application_service=self.vertex_model,
        #     persistence_service=s3_persistence_service,
        #     rag_chunker=rag_chunker,
        #     embeddings_manager=pg_embeddings_manager
        # )
        # context_chunks_in_document_service.delete_document_context_chunks(file_key)

class DeelabChromaChunksManager:

    def __init__(
            self,
            gcp_project_id: str,
            gcp_project_location: str,
            gcp_secret_name: str,
            llm_model_id: str = "claude-3-5-haiku@20241022",
            embeddings_model_id: str = "text-multilingual-embedding-002",
            target_language: str = "es",
            **chroma_conn_kwargs
    ):
        self.gcp_project_id = gcp_project_id
        self.gcp_project_location = gcp_project_location
        self.aws_secrets_manager = AwsSecretsManager()
        self.gcp_secret_name = gcp_secret_name
        self.llm_model_id = llm_model_id
        self.target_language = target_language
        self.gcp_sa_dict = self._get_gcp_sa_dict(gcp_secret_name)
        self.chroma_conn_kwargs = chroma_conn_kwargs
        self.vertex_model = self._get_vertex_model()
        self.embeddings_model = self.vertex_model.load_embeddings_model(embeddings_model_id)

    def _get_gcp_sa_dict(self, gcp_secret_name: str):
        vertex_gcp_sa = self.aws_secrets_manager.get_secret(gcp_secret_name)
        vertex_gcp_sa_dict = json.loads(vertex_gcp_sa)
        return vertex_gcp_sa_dict

    def _get_vertex_model(self):
        vertex_model = VertexModels(
            self.gcp_project_id,
            self.gcp_project_location,
            self.gcp_sa_dict,
            llm_model_id=self.llm_model_id
        )
        return vertex_model

    def context_chunks_in_document(
        self,
        file_key: str
    ):
        try:
            rag_chunker = SemanticChunks(self.embeddings_model)
            chroma_embeddings_manager = ChromaEmbeddingsManager(
                self.embeddings_model,
                {
                    "file_key": f"{file_key}"
                },
                **self.chroma_conn_kwargs
            )
            local_persistence_service = LocalStorageService()
            context_chunks_in_document_service = ContextChunksInDocumentService(
                ai_application_service=self.vertex_model,
                persistence_service=local_persistence_service,
                rag_chunker=rag_chunker,
                embeddings_manager=chroma_embeddings_manager,
                target_language=self.target_language
            )
            context_chunks = context_chunks_in_document_service.get_context_chunks_in_document(file_key)
            print("context_chunks", context_chunks)
            return context_chunks
        except Exception as e:
            print(f"Error getting context chunks in document: {e}")
            raise e

    # TODO
    def context_chunks_in_document_from_aws_cloud(
            self,
            file_key: str,
            s3_origin_bucket_name: str,
            s3_target_bucket_name: str
        ):
        try:
            s3_persistence_service = S3StorageService(
                origin_bucket_name=s3_origin_bucket_name,
                target_bucket_name=s3_target_bucket_name
            )
            target_bucket_file_tags = s3_persistence_service.retrieve_file_tags(file_key, s3_target_bucket_name)
            rag_chunker = SemanticChunks(self.embeddings_model)
            chroma_embeddings_manager = ChromaEmbeddingsManager(
                self.embeddings_model,
                {
                    "file_key": file_key
                },
                **self.chroma_conn_kwargs
            )
            context_chunks_in_document_service = ContextChunksInDocumentService(
                ai_application_service=self.vertex_model,
                persistence_service=s3_persistence_service,
                rag_chunker=rag_chunker,
                embeddings_manager=chroma_embeddings_manager,
                target_language=self.target_language
            )
            context_chunks = context_chunks_in_document_service.get_context_chunks_in_document(file_key, target_bucket_file_tags)
            return context_chunks
        except Exception as e:
            print(f"Error getting context chunks in document: {e}")
            raise e


    def delete_document_context_chunks_from_aws_cloud(
            self,
            file_key: str,
            s3_origin_bucket_name: str,
            s3_target_bucket_name: str
        ):
        pass
        # rag_chunker = SemanticChunks(self.embeddings_model)
        # pg_embeddings_manager = PgEmbeddingsManager(
        #     embeddings_model=self.embeddings_model,
        #     pg_connection=self.vector_store_connection
        # )
        # s3_persistence_service = S3StorageService(
        #     origin_bucket_name=s3_origin_bucket_name,
        #     target_bucket_name=s3_target_bucket_name
        # )
        # context_chunks_in_document_service = ContextChunksInDocumentService(
        #     ai_application_service=self.vertex_model,
        #     persistence_service=s3_persistence_service,
        #     rag_chunker=rag_chunker,
        #     embeddings_manager=pg_embeddings_manager
        # )
        # context_chunks_in_document_service.delete_document_context_chunks(file_key)
