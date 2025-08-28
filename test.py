import os
from dotenv import load_dotenv

load_dotenv()

from wizit_context_ingestor import  DeelabTranscribeManager, DeelabRedisChunksManager

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_PROJECT_LOCATION = os.environ.get("GCP_PROJECT_LOCATION")
SUPABASE_TABLE: str = os.environ.get("SUPABASE_TABLE")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
VECTOR_STORE_CONNECTION = os.environ.get("VECTOR_STORE_CONNECTION")
VECTOR_STORE_TABLE = os.environ.get("VECTOR_STORE_TABLE")
S3_ORIGIN_BUCKET_NAME = os.environ.get("S3_ORIGIN_BUCKET_NAME")
S3_TARGET_BUCKET_NAME = os.environ.get("S3_TARGET_BUCKET_NAME")
REDIS_CONNECTION_STRING = os.environ.get("REDIS_CONNECTION_STRING")


gcp_sa_path = os.path.join(os.path.dirname(__file__), "credentials", "gcp_sa.json")

if __name__ == '__main__':

    db_connection_secret_name = "tbbc-mega-ingestor-db-conn"
    gcp_secret_name = "tbbc-mega-ingestor-gcp-sa"

    # deelab_chunks_manager = DeelabChunksManager(
    #     GCP_PROJECT_ID,
    #     GCP_PROJECT_LOCATION,
    #     gcp_secret_name,
    #     db_connection_secret_name
    # )

    # deelab_transcribe_manager = DeelabTranscribeManager(
    #     GCP_PROJECT_ID,
    #     GCP_PROJECT_LOCATION,
    #     gcp_secret_name
    # )

    # deelab_transcribe_manager.aws_cloud_transcribe_document(
    #     "TBBC-2025.pdf",
    #     S3_ORIGIN_BUCKET_NAME,
    #     S3_TARGET_BUCKET_NAME
    # )

    deelab_chunks_manager = DeelabRedisChunksManager(
        GCP_PROJECT_ID,
        GCP_PROJECT_LOCATION,
        gcp_secret_name,
        REDIS_CONNECTION_STRING
    )

    deelab_chunks_manager.context_chunks_in_document_from_aws_cloud(
        "TBBC-2025.pdf.md",
        S3_ORIGIN_BUCKET_NAME,
        S3_TARGET_BUCKET_NAME
    )


    # deelab_chunks_manager.delete_document_context_chunks_from_aws_cloud(
    #     "TBBC-2025.pdf.md",
    #     S3_ORIGIN_BUCKET_NAME,
    #     S3_TARGET_BUCKET_NAME
    # )
