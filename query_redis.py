from langchain_redis import RedisConfig, RedisVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from redisvl.query.filter import Tag

load_dotenv()

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_PROJECT_LOCATION = os.environ.get("GCP_PROJECT_LOCATION")
S3_ORIGIN_BUCKET_NAME = os.environ.get("S3_ORIGIN_BUCKET_NAME")
S3_TARGET_BUCKET_NAME = os.environ.get("S3_TARGET_BUCKET_NAME")
REDIS_CONNECTION_STRING = os.environ.get("REDIS_CONNECTION_STRING")


gcp_sa_path = os.path.join(os.path.dirname(__file__), "credentials", "gcp_sa.json")
credentials = service_account.Credentials.from_service_account_file(gcp_sa_path)
# Initialize the embeddings model
embeddings_model = VertexAIEmbeddings(
    model_name="text-multilingual-embedding-002",
    credentials= credentials
)

# Initialize the Redis configuration
redis_config = RedisConfig(
index_name="vector_store",
redis_url="redis://localhost:6379",
metadata_schema=[
    {"type": "text", "name": "context"},
    {"type": "tag", "name": "id"}
],
)

# Initialize the Redis vector store
vector_store = RedisVectorStore(embeddings_model, config=redis_config)


if __name__ == "__main__":
    # Add documents to the Redis vector store
    query = "Como potencia la infra en la nube?"
    filter_condition = Tag("id") == "id"
    results = vector_store.similarity_search_with_score(query, k=2, filter=filter_condition)
    print(len(results))
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")
