import os
from dotenv import load_dotenv
from src.wizit_context_ingestor import ChunksManager, TranscriptionManager

# from src.wizit_context_ingestor.infra.persistence import LocalStorageService
import pyinstrument
import sys

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
        file_name = sys.argv[2]

        if file_name is None:
            file_name = "TBBC-2025.pdf.md"

        if operation == "transcribe":
            # storage_service = LocalStorageService("data", "tmp")

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
            )

            # deelab_transcribe_manager.aws_cloud_transcribe_document(
            #     file_name,
            #     S3_ORIGIN_BUCKET_NAME,
            #     S3_TARGET_BUCKET_NAME
            # )
            # deelab_transcribe_manager.transcribe_document(
            #     file_name, "s3", S3_ORIGIN_BUCKET_NAME, S3_TARGET_BUCKET_NAME
            # )
            deelab_transcribe_manager.transcribe_document(file_name)

        elif operation == "context":
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
            deelab_chunks_manager = ChunksManager(
                GCP_PROJECT_ID,
                GCP_PROJECT_LOCATION,
                gcp_secret_name,
                LANGSMITH_API_KEY,
                LANGCHAIN_PROJECT,
                "local",
                "chroma",
                {
                    "chroma_cloud_api_key": CHROMA_CLOUD_API_KEY,
                    "tenant": CHROMA_CLOUD_TENANT,
                    "database": CHROMA_COLLECTION_NAME,
                },
            )

            deelab_chunks_manager.gen_context_chunks(file_name, "tmp", "tmp")

            # deelab_chunks_manager.gen_context_chunks(
            #     file_name, S3_ORIGIN_BUCKET_NAME, S3_TARGET_BUCKET_NAME
            # )

    # execution examples
    # python test_redis.py transcribe TBBC-2025.pdf
    # python test_redis.py context GenAI-TBBC.pdf.md
