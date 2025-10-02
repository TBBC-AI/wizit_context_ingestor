import os
from dotenv import load_dotenv
from src.wizit_context_ingestor import (
    DeelabTranscribeManager,
    DeelabChromaChunksManager,
)
import pyinstrument
import sys

load_dotenv()

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_PROJECT_LOCATION = os.environ.get("GCP_PROJECT_LOCATION", "")
S3_ORIGIN_BUCKET_NAME = os.environ.get("S3_ORIGIN_BUCKET_NAME", "")
S3_TARGET_BUCKET_NAME = os.environ.get("S3_TARGET_BUCKET_NAME", "")
CHROMA_HOST = os.environ.get("REDIS_CONNECTION_STRING", "")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "")
CHROMA_CLOUD_API_KEY = os.environ.get("CHROMA_CLOUD_API_KEY", "")
CHROMA_CLOUD_TENANT = os.environ.get("CHROMA_CLOUD_TENANT", "")

print(S3_ORIGIN_BUCKET_NAME)
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
            deelab_transcribe_manager = DeelabTranscribeManager(
                GCP_PROJECT_ID,
                GCP_PROJECT_LOCATION,
                gcp_secret_name,
                transcription_additional_instructions="""
                    - HIGHLIGHTED CONTENT DETECTION:\n
                        - Wrap all highlighted content with <highlighted_content> tags.\n
                        - For tables with highlighted content, only column names must be wrapped in <highlighted_content> tags.\n
                        - Maintain the original order and formatting of the content.
                """,
            )

            deelab_transcribe_manager.aws_cloud_transcribe_document(
                file_name, S3_ORIGIN_BUCKET_NAME, S3_TARGET_BUCKET_NAME
            )
        elif operation == "context":
            if not file_name.endswith(".md"):
                raise ValueError("File name must be a markdown file")

            deelab_chunks_manager = DeelabChromaChunksManager(
                GCP_PROJECT_ID,
                GCP_PROJECT_LOCATION,
                gcp_secret_name,
                chroma_cloud_api_key=CHROMA_CLOUD_API_KEY,
                tenant=CHROMA_CLOUD_TENANT,
                database=CHROMA_COLLECTION_NAME,
            )

            deelab_chunks_manager.context_chunks_in_document_from_aws_cloud(
                file_name, S3_ORIGIN_BUCKET_NAME, S3_TARGET_BUCKET_NAME
            )

    # execution examples
    # python test_redis.py transcribe TBBC-2025.pdf
    # python test_redis.py context GenAI-TBBC.pdf.md
