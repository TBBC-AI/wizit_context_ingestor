import asyncio
import os
import sys

# from src.wizit_context_ingestor.infra.persistence import LocalStorageService
import pyinstrument
from dotenv import load_dotenv

from src.wizit_context_ingestor import (
    ChunksManager,
    PgKdbProvisioningManager,
    TranscriptionManager,
)

load_dotenv()

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_PROJECT_LOCATION = os.environ.get("GCP_PROJECT_LOCATION")
S3_ORIGIN_BUCKET_NAME = os.environ.get("S3_ORIGIN_BUCKET_NAME")
S3_TARGET_BUCKET_NAME = os.environ.get("S3_TARGET_BUCKET_NAME")
REDIS_CONNECTION_STRING = os.environ.get("REDIS_CONNECTION_STRING")
CHROMA_HOST = os.environ.get("REDIS_CONNECTION_STRING", "")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "")
CHROMA_CLOUD_API_KEY = os.environ.get("CHROMA_CLOUD_API_KEY", "")
CHROMA_CLOUD_TENANT = os.environ.get("CHROMA_CLOUD_TENANT", "")
PG_CONNECTION = os.environ.get("PG_CONNECTION", "")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "")
gcp_sa_path = os.path.join(os.path.dirname(__file__), "credentials", "gcp_sa.json")

if __name__ == "__main__":
    with pyinstrument.profile():
        # db_connection_secret_name = "tbbc-mega-ingestor-db-conn"
        gcp_secret_name = "tbbc-mega-ingestor-gcp-sa"

        if len(sys.argv) < 2:
            print("Please provide a file name as argument")
            sys.exit(1)

        operation = sys.argv[1]
        # file_name = sys.argv[2]

        # if file_name is None:
        #     file_name = "TBBC-2025.pdf.md"

        if operation == "transcribe":
            # storage_service = LocalStorageService("data", "tmp")
            file_name = sys.argv[2]

            if file_name is None:
                file_name = "TBBC-2025.pdf.md"

            deelab_transcribe_manager = TranscriptionManager(
                GCP_PROJECT_ID,
                GCP_PROJECT_LOCATION,
                gcp_secret_name,
                LANGSMITH_API_KEY,
                "test_wrapper",
                storage_service="local",
                source_storage_route="data",
                target_storage_route="tmp",
                transcription_additional_instructions="""
                    - HIGHLIGHTED CONTENT DETECTION:\n
                        - Wrap all highlighted content with <highlighted_content> tags.\n
                        - For tables with highlighted content, only column names must be wrapped in <highlighted_content> tags.\n
                        - Maintain the original order and formatting of the content.
                """,
                max_transcription_retries=1,
                transcription_accuracy_threshold=0.80,
            )

            # deelab_transcribe_manager.aws_cloud_transcribe_document(
            #     file_name,
            #     S3_ORIGIN_BUCKET_NAME,
            #     S3_TARGET_BUCKET_NAME
            # )
            # deelab_transcribe_manager.transcribe_document(
            #     file_name, "s3", S3_ORIGIN_BUCKET_NAME, S3_TARGET_BUCKET_NAME
            # )
            asyncio.run(deelab_transcribe_manager.transcribe_document(file_name))

        elif operation == "context":
            file_name = sys.argv[2]

            if file_name is None:
                file_name = "TBBC-2025.pdf.md"

            if not file_name.endswith(".md"):
                raise ValueError("File name must be a markdown file")

            # CHROMA PARAMETERS
            # chroma_cloud_api_key=CHROMA_CLOUD_API_KEY,
            # tenant=CHROMA_CLOUD_TENANT,
            # database=CHROMA_COLLECTION_NAME,
            # "redis_conn_string": REDIS_CONNECTION_STRING,
            # "chroma_cloud_api_key": CHROMA_CLOUD_API_KEY,
            # "tenant": CHROMA_CLOUD_TENANT,
            # "database": CHROMA_COLLECTION_NAME,
            # text-multilingual-embedding-002
            # postgresql://postgres.mitczeqkurkhkfsapeyh:[YOUR-PASSWORD]@aws-1-us-east-2.pooler.supabase.com:6543/postgres
            print(PG_CONNECTION)
            deelab_chunks_manager = ChunksManager(
                GCP_PROJECT_ID,
                GCP_PROJECT_LOCATION,
                gcp_secret_name,
                LANGSMITH_API_KEY,
                LANGCHAIN_PROJECT,
                "local",
                "pg",
                {
                    "pg_connection": PG_CONNECTION,
                    "embeddings_vectors_table_name": "otobotest",
                    "records_manager_table_name": "otobotest",
                    "content_column": "document",
                    "metadata_json_column": "metadata",
                    "id_column": "id",
                    "vector_size": 3072,
                },
                embeddings_model_id="gemini-embedding-001",
            )

            # deelab_chunks_manager.provision_vector_store()
            chunks = asyncio.run(
                deelab_chunks_manager.gen_context_chunks(file_name, "tmp", "tmp")
            )
            deelab_chunks_manager.index_documents_in_vector_store(chunks)
        elif operation == "query":
            query = sys.argv[2]

            if query is None:
                raise ValueError("Query cannot be None")

            deelab_chunks_manager = ChunksManager(
                GCP_PROJECT_ID,
                GCP_PROJECT_LOCATION,
                gcp_secret_name,
                LANGSMITH_API_KEY,
                LANGCHAIN_PROJECT,
                "local",
                "pg",
                {
                    "pg_connection": PG_CONNECTION,
                    "embeddings_vectors_table_name": "hdi",
                    "records_manager_table_name": "upsertion_record",
                    "content_column": "document",
                    "metadata_json_column": "metadata",
                    "id_column": "id",
                    "vector_size": 3072,
                },
                embeddings_model_id="gemini-embedding-001",
            )
            result = deelab_chunks_manager.search_records(query)
        elif operation == "provisioning":
            vector_store_name = sys.argv[2]

            if vector_store_name is None:
                raise ValueError("vector_store_name cannot be None")

            pg_kdb_provisioning_manager = PgKdbProvisioningManager(
                GCP_PROJECT_ID,
                GCP_PROJECT_LOCATION,
                gcp_secret_name,
                "gemini-embedding-001",
                {
                    "pg_connection": PG_CONNECTION,
                    "embeddings_vectors_table_name": vector_store_name,
                    "records_manager_table_name": vector_store_name,
                    "content_column": "document",
                    "metadata_json_column": "metadata",
                    "id_column": "id",
                    "vector_size": 3072,
                },
            )
            pg_kdb_provisioning_manager.provision_vector_store()

        elif operation == "find_by_name":
            file_name = sys.argv[2]

            if file_name is None:
                raise ValueError("file_name cannot be None")

            deelab_chunks_manager = ChunksManager(
                GCP_PROJECT_ID,
                GCP_PROJECT_LOCATION,
                gcp_secret_name,
                LANGSMITH_API_KEY,
                LANGCHAIN_PROJECT,
                "local",
                "pg",
                {
                    "pg_connection": PG_CONNECTION,
                    "embeddings_vectors_table_name": "otobotest",
                    "records_manager_table_name": "otobotest",
                    "content_column": "document",
                    "metadata_json_column": "metadata",
                    "id_column": "id",
                    "vector_size": 3072,
                },
                embeddings_model_id="gemini-embedding-001",
            )
            retrieved_docs = deelab_chunks_manager.search_records_by_file_name(
                file_name
            )
            print(len(retrieved_docs))

        elif operation == "delete_by_name":
            file_name = sys.argv[2]

            if file_name is None:
                raise ValueError("file_name cannot be None")

            deelab_chunks_manager = ChunksManager(
                GCP_PROJECT_ID,
                GCP_PROJECT_LOCATION,
                gcp_secret_name,
                LANGSMITH_API_KEY,
                LANGCHAIN_PROJECT,
                "local",
                "pg",
                {
                    "pg_connection": PG_CONNECTION,
                    "embeddings_vectors_table_name": "otobotest",
                    "records_manager_table_name": "otobotest",
                    "content_column": "document",
                    "metadata_json_column": "metadata",
                    "id_column": "id",
                    "vector_size": 3072,
                },
                embeddings_model_id="gemini-embedding-001",
            )
            deleted_docs = deelab_chunks_manager.delete_documents_by_file_name(
                file_name
            )
            print(len(deleted_docs))
