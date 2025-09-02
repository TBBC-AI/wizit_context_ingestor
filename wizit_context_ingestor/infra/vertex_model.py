import vertexai
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from ..data.prompts import \
    IMAGE_TRANSCRIPTION_SYSTEM_PROMPT, \
    CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT, \
    ContextChunk
from typing import Dict, Any, Optional, List, Union
from ..application.interfaces import AiApplicationService
from ..domain.models import ParsedDocPage
import logging


logger = logging.getLogger(__name__)


class VertexModels(AiApplicationService):
    """
    A wrapper class for Google Cloud Vertex AI models that handles credentials and
    provides methods to load embeddings and chat models.
    """

    def __init__(
            self,
            project_id: str,
            location: str,
            json_service_account: Dict[str, Any],
            scopes: Optional[List[str]] = None,
            llm_model_id: str = "claude-3-5-haiku@20241022"):
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
                json_service_account,
                scopes=self.scopes
            )
            self.llm_model_id = llm_model_id
            self.project_id = project_id
            self.location = location
            vertexai.init(
                project=project_id,
                location=location,
                credentials=self.credentials
            )
            logger.info(f"VertexModels initialized with project {project_id} in {location}")
        except Exception as e:
            logger.error(f"Failed to initialize VertexModels: {str(e)}")
            raise

    def load_embeddings_model(
        self,
        embeddings_model_id: str = "text-embedding-005") -> VertexAIEmbeddings:  # noqa: E125
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
            logger.error(f"Failed to load embeddings model {embeddings_model_id}: {str(e)}")
            raise

    def load_chat_model(self,
        temperature: float = 0.15,
        max_tokens: int = 8192,
        stop: Optional[List[str]] = None,
        **chat_model_params) -> ChatVertexAI:
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
                return self.load_chat_model_gemini(self.llm_model_id, temperature, max_tokens, stop, **chat_model_params)
            elif "claude" in self.llm_model_id:
                return self.load_chat_model_anthropic(self.llm_model_id, temperature, max_tokens, stop, **chat_model_params)
            else:
                raise ValueError(f"Unsupported chat model: {self.llm_model_id}")
        except Exception as e:
            logger.error(f"Failed to retrieve chat model {self.llm_model_id}: {str(e)}")
            raise

    def load_chat_model_gemini(self,
        chat_model_id: str = "publishers/google/models/gemini-2.5-flash",
        temperature: float = 0.15,
        max_tokens: int = 8192,
        stop: Optional[List[str]] = None,
        **chat_model_params) -> ChatVertexAI:
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
                **chat_model_params
            )
            logger.debug(f"Retrieved chat model: {chat_model_id}")
            return self.llm_model
        except Exception as e:
            logger.error(f"Failed to retrieve chat model {chat_model_id}: {str(e)}")
            raise

    def load_chat_model_anthropic(self,
        chat_model_id: str = "claude-3-5-haiku@20241022",
        temperature: float = 0.7,
        max_tokens: int = 8000,
        stop: Optional[List[str]] = None,
        **chat_model_params) -> ChatAnthropicVertex:
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
                **chat_model_params
            )
            logger.debug(f"Retrieved chat model: {chat_model_id}")
            return self.llm_model
        except Exception as e:
            logger.error(f"Failed to retrieve chat model {chat_model_id}: {str(e)}")
            raise

    def parse_doc_page(self, document: ParsedDocPage) -> ParsedDocPage:
        """Transcribe an image to text.

        Args:
            document: The document with the image to transcribe

        Returns:
            Processed text
        """
        try:
            output_parser = StrOutputParser()
            # Create the prompt template with image
            prompt = ChatPromptTemplate.from_messages([
                ("system", IMAGE_TRANSCRIPTION_SYSTEM_PROMPT),
                ("human", [{
                        "type": "image",
                        "image_url": {
                            "url": f"data:image/png;base64,{document.page_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Transcribe the image"
                    }]
                ),
            ])
            # Create the chain
            chain = prompt | self.llm_model | output_parser
            # Process the image
            result = chain.invoke({})
            document.page_text = result
            return document
        except Exception as e:
            logger.error(f"Failed to parse document page: {str(e)}")
            raise

    def _retrieve_context_chunk_in_document(self, markdown_content: str, chunk: Document, chunk_metadata: dict = None) -> ContextChunk:
        """Retrieve context chunks in document."""
        try:
            chunk_output_parser = PydanticOutputParser(pydantic_object=ContextChunk)
            # Create the prompt template with image
            prompt = ChatPromptTemplate.from_messages([
                ("system", CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT),
                (
                    "human", [{
                        "type": "text",
                            "text": f"Generate context for the following chunk: <chunk>{chunk.page_content}</chunk>"
                    }]
                ),
            ]).partial(
                document_content=markdown_content,
                format_instructions=chunk_output_parser.get_format_instructions()
            )
            model_with_structure = self.llm_model.with_structured_output(ContextChunk)
            # Create the chain
            chain = prompt | model_with_structure
            # Process the image
            results = chain.invoke({})
            print(chunk)
            chunk.page_content = f"Context:{results.context}, Content:{chunk.page_content}"
            chunk.metadata["context"] = results.context
            if chunk_metadata:
                for key, value in chunk_metadata.items():
                    chunk.metadata[key] = value
            return chunk

        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
            raise


    def retrieve_context_chunks_in_document(self, markdown_content: str, chunks: List[Document], chunks_metadata: dict = None) -> List[Document]:
        """Retrieve context chunks in document."""
        try:
            context_chunks = list(map(
                lambda chunk: self._retrieve_context_chunk_in_document(markdown_content, chunk, chunks_metadata),
                chunks
            ))
            return context_chunks
        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
            raise

    # @contextmanager
    # def model_context(self):
    #     """
    #     Context manager for VertexModels to ensure proper resource cleanup.

    #     Example:
    #         with vertex_models.model_context():
    #             # Use vertex models here
    #     """
    #     try:
    #         yield self
    #     finally:
    #         # Clean up any resources if needed
    #         # This can be expanded based on specific cleanup requirements
    #         logger.debug("Exiting VertexModels context")
