import logging

from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import StdOutCallbackHandler
from typing_extensions import List, Optional

from ..application.interfaces import AiApplicationService

logger = logging.getLogger(__name__)


class AWSModels(AiApplicationService):
    """
    A wrapper class for Google Cloud Vertex AI models that handles credentials and
    provides methods to load embeddings and chat models.
    """

    __slots__ = "llm_model_id"

    def __init__(
        self, llm_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    ):
        """
        Initialize the VertexModels class with Google Cloud credentials.

        Args:
            project_id: The Google Cloud project ID
            location: The Google Cloud region (e.g., "us-central1")
            json_service_account: Dictionary containing service account credentials
            scopes: Optional list of authentication scopes. Defaults to cloud platform scope.
        """
        self.llm_model_id = llm_model_id

    def load_embeddings_model(self):  # noqa: E125
        raise NotImplementedError("Not implemented")

    #     self,
    #     temperature: float = 0.15,
    #     max_tokens: int = 20000,
    #     stop: Optional[List[str]] = None,
    #     **chat_model_params,
    # ) -> Union[ChatVertexAI, ChatAnthropicVertex]:

    def load_chat_model(
        self,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        stop: Optional[List[str]] = None,
        **aws_params,
    ) -> ChatBedrockConverse:
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
            # if aws_params["region_name"]:
            #     region_name = aws_params["region_name"]
            # else:
            #     region_name = "us-east-1"

            self.llm_model = ChatBedrockConverse(
                model=self.llm_model_id,
                temperature=temperature,
                callbacks=[StdOutCallbackHandler()],
                max_tokens=max_tokens,
                region_name=aws_params["region_name"]
                if aws_params["region_name"]
                else "us-east-1",
            )
            logging.info("model activated")
            return self.llm_model
        except Exception as error:
            logging.error(f"Error to retrieve chat model: {str(error)}")
            raise error
