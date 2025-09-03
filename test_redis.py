import os
from dotenv import load_dotenv
from wizit_context_ingestor import DeelabRedisChunksManager, DeelabTranscribeManager
import sys

load_dotenv()

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_PROJECT_LOCATION = os.environ.get("GCP_PROJECT_LOCATION")
S3_ORIGIN_BUCKET_NAME = os.environ.get("S3_ORIGIN_BUCKET_NAME")
S3_TARGET_BUCKET_NAME = os.environ.get("S3_TARGET_BUCKET_NAME")
REDIS_CONNECTION_STRING = os.environ.get("REDIS_CONNECTION_STRING")

gcp_sa_path = os.path.join(os.path.dirname(__file__), "credentials", "gcp_sa.json")

if __name__ == '__main__':
    # db_connection_secret_name = "tbbc-mega-ingestor-db-conn"
    gcp_secret_name = "tbbc-mega-ingestor-gcp-sa"

    if len(sys.argv) < 2:
        print("Please provide a file name as argument")
        sys.exit(1)

    operation = sys.argv[1]
    file_name = sys.argv[2]

    if file_name == None:
        file_name = "TBBC-2025.pdf.md"

    if operation == "transcribe":
        deelab_transcribe_manager = DeelabTranscribeManager(
            GCP_PROJECT_ID,
            GCP_PROJECT_LOCATION,
            gcp_secret_name
        )

        deelab_transcribe_manager.aws_cloud_transcribe_document(
            file_name,
            S3_ORIGIN_BUCKET_NAME,
            S3_TARGET_BUCKET_NAME
        )
    elif operation == "context":

        if not file_name.endswith(".md"):
            raise ValueError("File name must be a markdown file")

        deelab_chunks_manager = DeelabRedisChunksManager(
            GCP_PROJECT_ID,
            GCP_PROJECT_LOCATION,
            gcp_secret_name,
            REDIS_CONNECTION_STRING
        )
        deelab_chunks_manager.context_chunks_in_document_from_aws_cloud(
            file_name,
            S3_ORIGIN_BUCKET_NAME,
            S3_TARGET_BUCKET_NAME
        )

# execution examples
# python test_redis.py transcribe TBBC-2025.pdf
# python test_redis.py context GenAI-TBBC.pdf.md
