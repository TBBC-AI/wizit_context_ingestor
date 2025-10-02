from vertexai import init as vertexai_init
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from typing import Dict, Any, Optional, List, Union
from ..application.interfaces import AiApplicationService
import logging


logger = logging.getLogger(__name__)


class VertexModels(AiApplicationService):
    """
    A wrapper class for Google Cloud Vertex AI models that handles credentials and
    provides methods to load embeddings and chat models.
    """

    __slots__ = (
        "project_id",
        "location",
        "json_service_account",
        "scopes",
        "llm_model_id",
    )

    def __init__(
        self,
        project_id: str,
        location: str,
        json_service_account: Dict[str, Any],
        scopes: Optional[List[str]] = None,
        llm_model_id: str = "claude-sonnet-4@20250514",
    ):
        """
        Initialize the VertexModels class with Google Cloud credentials.

        Args:
            project_id: The Google Cloud project ID
            location: The Google Cloud region (e.g., "us-central1")
            json_service_account: Dictionary containing service account credentials
            scopes: Optional list of authentication scopes. Defaults to cloud platform scope.
        """
        try:
            print(location)
            self.scopes = scopes or ["https://www.googleapis.com/auth/cloud-platform"]
            self.credentials = service_account.Credentials.from_service_account_info(
                json_service_account, scopes=self.scopes
            )
            self.llm_model_id = llm_model_id
            self.project_id = project_id
            self.location = location
            vertexai_init(
                project=project_id, location=location, credentials=self.credentials
            )
            logger.info(
                f"VertexModels initialized with project {project_id} in {location}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize VertexModels: {str(e)}")
            raise

    def load_embeddings_model(
        self, embeddings_model_id: str = "text-multilingual-embedding-002"
    ) -> VertexAIEmbeddings:  # noqa: E125
        """
        Load and return a Vertex AI embeddings model.
        default embeddings length is 768 https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
        Args:
            embeddings_model_id: The ID of the embedding model to use.
                                Default is "text-embedding-005".

        Returns:
            An instance of VertexAIEmbeddings ready for generating embeddings.
        """
        try:
            embeddings = VertexAIEmbeddings(
                model=embeddings_model_id,
                credentials=self.credentials,
            )
            logger.debug(f"Loaded embedding model: {embeddings_model_id}")
            return embeddings
        except Exception as e:
            logger.error(
                f"Failed to load embeddings model {embeddings_model_id}: {str(e)}"
            )
            raise

    def load_chat_model(
        self,
        temperature: float = 0.15,
        max_tokens: int = 8192,
        stop: Optional[List[str]] = None,
        **chat_model_params,
    ) -> Union[ChatVertexAI, ChatAnthropicVertex]:
        """
        Load a Vertex AI chat model for text generation.

        Args:
            chat_model_id: The ID of the chat model to use.
                         Default is "gemini-1.5-flash-001".
            temperature: Controls randomness in responses. Lower values make responses
                        more deterministic. Default is 0.1.
            max_tokens: Maximum number of tokens to generate. Default is 8192.
            stop: Optional list of strings that will stop generation when encountered.
            **chat_model_params: Additional parameters to pass to the chat model.

        Returns:
            An instance of ChatVertexAI ready for chat interactions.
        """
        try:
            if "gemini" in self.llm_model_id:
                return self.load_chat_model_gemini(
                    self.llm_model_id,
                    temperature,
                    max_tokens,
                    stop,
                    **chat_model_params,
                )
            elif "claude" in self.llm_model_id:
                return self.load_chat_model_anthropic(
                    self.llm_model_id,
                    temperature,
                    max_tokens,
                    stop,
                    **chat_model_params,
                )
            else:
                raise ValueError(f"Unsupported chat model: {self.llm_model_id}")
        except Exception as e:
            logger.error(f"Failed to retrieve chat model {self.llm_model_id}: {str(e)}")
            raise

    def load_chat_model_gemini(
        self,
        chat_model_id: str = "publishers/google/models/gemini-2.5-flash",
        temperature: float = 0.15,
        max_tokens: int = 64000,
        stop: Optional[List[str]] = None,
        **chat_model_params,
    ) -> ChatVertexAI:
        """
        Load a Vertex AI chat model for text generation.

        Args:
            chat_model_id: The ID of the chat model to use.
                         Default is "gemini-1.5-flash-001".
            temperature: Controls randomness in responses. Lower values make responses
                        more deterministic. Default is 0.1.
            max_tokens: Maximum number of tokens to generate. Default is 8192.
            stop: Optional list of strings that will stop generation when encountered.
            **chat_model_params: Additional parameters to pass to the chat model.

        Returns:
            An instance of ChatVertexAI ready for chat interactions.
        """
        try:
            self.llm_model = ChatVertexAI(
                model=chat_model_id,
                location=self.location,  # Use the same location as the project,
                temperature=temperature,
                credentials=self.credentials,
                max_tokens=max_tokens,
                max_retries=1,
                stop=stop,
                **chat_model_params,
            )
            logger.debug(f"Retrieved chat model: {chat_model_id}")
            return self.llm_model
        except Exception as e:
            logger.error(f"Failed to retrieve chat model {chat_model_id}: {str(e)}")
            raise

    def load_chat_model_anthropic(
        self,
        chat_model_id: str = "claude-3-5-haiku@20241022",
        temperature: float = 0.7,
        max_tokens: int = 64000,
        stop: Optional[List[str]] = None,
        **chat_model_params,
    ) -> ChatAnthropicVertex:
        """
        Load a Vertex AI chat model for text generation.
        """
        try:
            self.llm_model = ChatAnthropicVertex(
                model=chat_model_id,
                location=self.location,  # Use the same location as the project,
                temperature=temperature,
                credentials=self.credentials,
                max_tokens=max_tokens,
                max_retries=1,
                stop=stop,
                **chat_model_params,
            )
            logger.debug(f"Retrieved chat model: {chat_model_id}")
            return self.llm_model
        except Exception as e:
            logger.error(f"Failed to retrieve chat model {chat_model_id}: {str(e)}")
            raise
