from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import StdOutCallbackHandler
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


class AWSModels(AiApplicationService):
    """
    A wrapper class for Google Cloud Vertex AI models that handles credentials and
    provides methods to load embeddings and chat models.
    """

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
        raise "Not implemented"

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

    def _retrieve_context_chunk_in_document(self, markdown_content: str, chunk: Document) -> ContextChunk:
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
            # Create the chain
            chain = prompt | self.llm_model | chunk_output_parser
            # Process the image
            results = chain.invoke({})
            chunk.page_content = f"Context:{results.context}, Content:{chunk.page_content}"
            chunk.metadata["context"] = results.context
            return chunk

        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")


    def retrieve_context_chunks_in_document(self, markdown_content: str, chunks: List[Document]) -> List[Document]:
        """Retrieve context chunks in document."""
        try:
            context_chunks = list(map(
                lambda chunk: self._retrieve_context_chunk_in_document(markdown_content, chunk),
                chunks
            ))
            return context_chunks
        except Exception as e:
            logger.error(f"Failed to retrieve context chunks in document: {str(e)}")
            raise
