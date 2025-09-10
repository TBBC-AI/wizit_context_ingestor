from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import StdOutCallbackHandler
from ..application.interfaces import AiApplicationService
import logging


logger = logging.getLogger(__name__)


class AWSModels(AiApplicationService):
    """
    A wrapper class for Google Cloud Vertex AI models that handles credentials and
    provides methods to load embeddings and chat models.
    """
    __slots__ = ('llm_model_id')
    def __init__(
        self,
        llm_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    ):
        """
        Initialize the VertexModels class with Google Cloud credentials.

        Args:
            project_id: The Google Cloud project ID
            location: The Google Cloud region (e.g., "us-central1")
            json_service_account: Dictionary containing service account credentials
            scopes: Optional list of authentication scopes. Defaults to cloud platform scope.
        """
        print("Initializing AWS model")
        self.llm_model_id = llm_model_id

    def load_embeddings_model(self):  # noqa: E125
        raise NotImplementedError("Not implemented")

    def load_chat_model(
        self,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        region_name: str = "us-east-1") -> ChatBedrockConverse:
        """
        Load an AWS AI chat model for text generation.

        Args:
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            region_name=region_name

        Returns:
            An instance of ChatVertexAI ready for chat interactions.
        """
        try:
            self.llm_model = ChatBedrockConverse(
                model=self.llm_model_id,
                temperature=temperature,
                callbacks=[StdOutCallbackHandler()],
                max_tokens=max_tokens,
                region_name=region_name
            )
            # if self.is_external_provider:
            #     print("Usando credenciales externas")
            #     credentials = self.load_sts_credentials()
            #     bedrock_chat.aws_access_key_id=credentials['AccessKeyId']
            #     bedrock_chat.aws_secret_access_key=credentials['SecretAccessKey']
            #     bedrock_chat.aws_session_token=credentials['SessionToken']
            logging.info("model activated")
            return self.llm_model
        except Exception as error:
            logging.error(f"Error to retrieve chat model: {str(error)}")
            raise error
