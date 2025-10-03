from typing import Tuple, List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.messages import HumanMessage
from logging import getLogger
from ..data.prompts import IMAGE_TRANSCRIPTION_SYSTEM_PROMPT, Transcription
from ..domain.models import ParsedDoc, ParsedDocPage
from ..domain.services import ParseDocModelService
from .interfaces import AiApplicationService, PersistenceService
from ..workflows.transcription_workflow import TranscriptionWorkflow

logger = getLogger(__name__)


class TranscriptionService:
    """
    Service for transcribing documents.
    """

    def __init__(
        self,
        ai_application_service: AiApplicationService,
        persistence_service: PersistenceService,
        target_language: str = "es",
        transcription_additional_instructions: str = "",
        transcription_accuracy_threshold: int = 90,
        max_transcription_retries: int = 2,
    ):
        self.ai_application_service = ai_application_service
        self.persistence_service = persistence_service
        self.target_language = target_language
        if (
            transcription_accuracy_threshold < 0
            or transcription_accuracy_threshold > 95
        ):
            raise ValueError(
                "transcription_accuracy_threshold must be between 0 and 95"
            )
        if max_transcription_retries < 1 or max_transcription_retries > 3:
            raise ValueError(
                "max_transcription_retries must be between 1 and 3 to prevent token exhaustion"
            )
        self.transcription_accuracy_threshold = transcription_accuracy_threshold
        self.max_transcription_retries = max_transcription_retries
        self.transcription_additional_instructions = (
            transcription_additional_instructions
        )
        self.chat_model = self.ai_application_service.load_chat_model()

    # def parse_doc_page(self, document: ParsedDocPage) -> ParsedDocPage:
    #     """Transcribe an image to text.
    #     Args:
    #         document: The document with the image to transcribe
    #     Returns:
    #         Processed text
    #     """
    #     try:
    #         # Create the prompt template with image
    #         transcription_output_parser = PydanticOutputParser(
    #             pydantic_object=Transcription
    #         )
    #         prompt = ChatPromptTemplate.from_messages(
    #             [
    #                 ("system", IMAGE_TRANSCRIPTION_SYSTEM_PROMPT),
    #                 (
    #                     "human",
    #                     [
    #                         {
    #                             "type": "image",
    #                             "image_url": {
    #                                 "url": f"data:image/png;base64,{document.page_base64}"
    #                             },
    #                         },
    #                         {
    #                             "type": "text",
    #                             "text": "Transcribe the document, ensure all content transcribed accurately",
    #                         },
    #                     ],
    #                 ),
    #             ]
    #         ).partial(
    #             transcription_additional_instructions=self.transcription_additional_instructions,
    #             format_instructions=transcription_output_parser.get_format_instructions(),
    #         )
    #         model_with_structured_output = self.chat_model.with_structured_output(
    #             Transcription
    #         )
    #         # Create the chain
    #         chain = prompt | model_with_structured_output
    #         # Process the image
    #         chain = chain.with_retry(
    #             stop_after_attempt=3, exponential_jitter_params={"initial": 60}
    #         )
    #         result = chain.invoke({})
    #         if result.transcription:
    #             document.page_text = result.transcription
    #         else:
    #             raise ValueError("No transcription found")
    #         return document
    #     except Exception as e:
    #         logger.error(f"Failed to parse document page: {str(e)}")
    #         raise

    def parse_doc_page_with_workflow(self, document: ParsedDocPage) -> ParsedDocPage:
        """Transcribe an image to text using an agent.
        Args:
            document: The document with the image to transcribe
        Returns:
            Processed text
        """
        transcription_workflow = TranscriptionWorkflow(
            self.chat_model, self.transcription_additional_instructions
        )
        compiled_transcription_workflow = transcription_workflow.gen_workflow()
        compiled_transcription_workflow = compiled_transcription_workflow.compile()
        result = compiled_transcription_workflow.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Transcribe the document, ensure all content transcribed accurately. transcription must be in the same language of source document.",
                            },
                        ]
                    ),
                    HumanMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{document.page_base64}"
                                },
                            }
                        ]
                    ),
                ]
            },
            {
                "configurable": {
                    "transcription_accuracy_threshold": self.transcription_accuracy_threshold,
                    "max_transcription_retries": self.max_transcription_retries,
                }
            },
        )
        if result["transcription"]:
            document.page_text = result["transcription"]
        else:
            raise ValueError("No transcription found")
        return document

    def process_document(self, file_key: str) -> Tuple[List[ParsedDocPage], ParsedDoc]:
        """
        Process a document by parsing it and returning the parsed content.
        """
        raw_file_path = self.persistence_service.retrieve_raw_file(file_key)
        parse_doc_model_service = ParseDocModelService(raw_file_path)
        document_pages = parse_doc_model_service.parse_document_to_base64()
        parsed_pages = []
        for page in document_pages:
            page = self.parse_doc_page_with_workflow(page)
            parsed_pages.append(page)
        logger.info(f"Parsed {len(parsed_pages)} pages")
        parsed_document = parse_doc_model_service.create_md_content(parsed_pages)
        return parsed_pages, parsed_document

    def save_parsed_document(
        self,
        file_key: str,
        parsed_document: ParsedDoc,
        file_tags: Optional[Dict[str, str]] = {},
    ):
        """
        Save the parsed document to a file.
        """
        self.persistence_service.save_parsed_document(
            file_key, parsed_document, file_tags
        )
