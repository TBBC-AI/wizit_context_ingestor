from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_aws import ChatBedrockConverse
from logging import getLogger
from ..data.prompts import IMAGE_TRANSCRIPTION_SYSTEM_PROMPT
from ..domain.models import ParsedDocPage
from .interfaces import AiApplicationService

logger = getLogger(__name__)


class TranscriptionService:
    """
        Service for transcribing documents.
    """

    def __init__(self, ai_application_service: AiApplicationService):
        self.ai_application_service = ai_application_service
        self.chat_model = self.ai_application_service.load_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", IMAGE_TRANSCRIPTION_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history")
        ])

    def parse_doc_page(self, document: ParsedDocPage) -> ParsedDocPage:
            """Transcribe an image to text.
            Args:
                document: The document with the image to transcribe
            Returns:
                Processed text
            """
            try:
                # Create the chain
                chain = self.prompt | self.chat_model | StrOutputParser()
                # Process the image
                result = chain.invoke({
                    "history": [
                        {
                            "type": "text",
                            "text": "Transcribe the image"
                        },
                        {
                            "type": "image",
                            "image_url": {
                                "url": f"data:image/png;base64,{document.page_base64}"
                            }
                        }
                    ]
                })
                document.page_text = result
                return document
            except Exception as e:
                logger.error(f"Failed to parse document page: {str(e)}")
                raise
