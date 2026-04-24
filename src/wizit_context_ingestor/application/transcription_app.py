import asyncio
from logging import getLogger
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langsmith import Client, tracing_context

from ..domain.models import ParsedDoc, ParsedDocPage
from ..domain.services import ParseDocModelService
from ..workflows.transcription_workflow import TranscriptionWorkflow
from .interfaces import AiApplicationService, PersistenceService

logger = getLogger(__name__)


class TranscriptionApp:
    """
    Service for transcribing documents.
    """

    def __init__(
        self,
        ai_application_service: AiApplicationService,
        persistence_service: PersistenceService,
        langsmith_api_key: str,
        langsmith_project_name: str,
        target_language: str = "es",
        transcription_additional_instructions: str = "",
        transcription_accuracy_threshold: float = 0.90,
        max_transcription_retries: int = 2,
        **langsmith_config,
    ):
        self.ai_application_service = ai_application_service
        self.persistence_service = persistence_service
        self.target_language = target_language
        self.langsmith_project_name = langsmith_project_name
        if (
            transcription_accuracy_threshold < 0.0
            or transcription_accuracy_threshold > 0.95
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
        self.transcription_workflow = TranscriptionWorkflow(
            self.chat_model, self.transcription_additional_instructions
        )
        self.workflow = self.transcription_workflow.gen_workflow()
        self.compiled_transcription_workflow = self.workflow.compile()
        self.langsmith_client = Client(api_key=langsmith_api_key)

    async def parse_doc_page_with_workflow(
        self, document: ParsedDocPage, retries: int = 0
    ) -> ParsedDocPage:
        """Transcribe an image to text using an agent.
        Args:
            document: The document with the image to transcribe
        Returns:
            Processed text
        """
        if retries > 1:
            logger.info("Max retries exceeded")
            return document

        with tracing_context(
            enabled=True,
            project_name=self.langsmith_project_name,
            client=self.langsmith_client,
        ):
            result = await self.compiled_transcription_workflow.ainvoke(
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
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": document.page_base64,
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
            if "transcription" in result:
                document.page_text = result["transcription"]
            else:
                return await self.parse_doc_page_with_workflow(
                    document, retries=retries + 1
                )
            return document

    async def process_document(self, file_key: str) -> Tuple[ParsedDoc, dict]:
        """
        Process a document by parsing it and returning the parsed content.
        """
        raw_file_path, metadata = self.persistence_service.retrieve_raw_file(file_key)
        parse_doc_model_service = ParseDocModelService(raw_file_path)
        document_pages = parse_doc_model_service.parse_document_to_base64_pages()
        parse_pages_workflow_tasks = []
        parsed_pages = []
        for page in document_pages:
            parse_pages_workflow_tasks.append(self.parse_doc_page_with_workflow(page))
        # here
        parsed_pages = await asyncio.gather(*parse_pages_workflow_tasks)
        logger.info(f"Parsed {len(parsed_pages)} pages")
        parsed_document = parse_doc_model_service.create_md_content(parsed_pages)
        return parsed_document, metadata

    def save_parsed_document(
        self,
        file_key: str,
        parsed_document: ParsedDoc,
        file_tags: Optional[Dict[str, str]] = {},
        metadata: Optional[Dict[str, str]] = {},
    ):
        """
        Save the parsed document to a file.
        """
        self.persistence_service.save_parsed_document(
            file_key, parsed_document, file_tags, metadata
        )
