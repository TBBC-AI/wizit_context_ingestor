from typing import Tuple, List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from logging import getLogger
from ..data.prompts import IMAGE_TRANSCRIPTION_SYSTEM_PROMPT
from ..domain.models import ParsedDoc, ParsedDocPage
from ..domain.services import ParseDocModelService
from .interfaces import AiApplicationService, PersistenceService

logger = getLogger(__name__)


class TranscriptionService:
    """
        Service for transcribing documents.
    """

    def __init__(
        self,
        ai_application_service: AiApplicationService,
        persistence_service: PersistenceService,
        target_language: str = 'es'
    ):
        self.ai_application_service = ai_application_service
        self.persistence_service = persistence_service
        self.target_language = target_language
        self.chat_model = self.ai_application_service.load_chat_model()

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
                            "text": f"Transcribe the document, ensure all content transcribed is using '{self.target_language}' language"
                        }]
                    ),
                ])
                # Create the chain
                chain = prompt | self.chat_model | output_parser
                # Process the image
                result = chain.invoke({})
                document.page_text = result
                return document
            except Exception as e:
                logger.error(f"Failed to parse document page: {str(e)}")
                raise

    def process_document(self, file_key: str) -> Tuple[List[ParsedDocPage], ParsedDoc]:
        """
        Process a document by parsing it and returning the parsed content.
        """
        raw_file_path = self.persistence_service.retrieve_raw_file(file_key)
        parse_doc_model_service = ParseDocModelService(raw_file_path)
        document_pages = parse_doc_model_service.parse_document_to_base64()
        parsed_pages = []
        for page in document_pages:
            page = self.parse_doc_page(page)
            parsed_pages.append(page)
        logger.info(f"Parsed {len(parsed_pages)} pages")
        parsed_document = parse_doc_model_service.create_md_content(parsed_pages)
        return parsed_pages, parsed_document


    def save_parsed_document(self, file_key: str, parsed_document: ParsedDoc, file_tags: Optional[Dict[str, str]] = None):
        """
        Save the parsed document to a file.
        """
        self.persistence_service.save_parsed_document(file_key, parsed_document, file_tags)
