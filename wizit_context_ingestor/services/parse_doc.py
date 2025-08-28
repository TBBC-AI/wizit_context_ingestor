from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64
import logging
import io
import fitz
from PIL import Image
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# CHECK THIS THING IMPROVE THE WAY CODE IS STRUCTURED
class ParseDoc:
    """
    Class for parsing PDF documents, converting pages to base64 images,
    and processing them with language models.
    """

    def __init__(self, file_path: str, system_prompt, chat_model: Any):
        """
        Initialize a PDF document parser.
        
        Args:
            file_path: Path to the PDF file to parse
            chat_model: Language model for processing document content
        """
        self.file_path = file_path
        self.pdf_document = fitz.open(file_path)
        self.page_count = self.pdf_document.page_count
        self.system_prompt = system_prompt
        self.chat_model = chat_model

    def pdf_page_to_base64(self, page_number: int) -> str:
        """
        Convert a PDF page to a base64-encoded PNG image.
        
        Args:
            page_number: One-indexed page number to convert
            
        Returns:
            Base64 encoded string of the page image
            
        Raises:
            Exception: If there's an error during conversion
        """
        try:
            # input is one-indexed
            page = self.pdf_document.load_page(page_number - 1)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            png_data = buffer.getvalue()
            # Validate PNG header
            if not png_data.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError("Generated data is not a valid PNG")
            b64_encoded_image = base64.b64encode(png_data).decode("utf-8")
            logger.info(f"Page {page_number} encoded successfully")
            return b64_encoded_image
        except Exception as e:
            logger.error(f"Failed to parse b64 image: {str(e)}")
            raise

    def parse_document_to_base64(self) -> List[str]:
        """
        Convert all pages in the PDF document to base64-encoded images.
        
        Returns:
            List of base64 encoded strings for each page
            
        Raises:
            Exception: If there's an error during conversion
        """
        try:
            base64_pages = []
            for page_number in range(1, self.pdf_document.page_count + 1):
                base64_image = self.pdf_page_to_base64(page_number)
                base64_pages.append(base64_image)
            logger.info(f"{len(base64_pages)} Pages encoded to base64 successfully")
            return base64_pages
        except Exception as e:
            logger.error(f"Failed to parse b64 image: {str(e)}")
            raise

    def parse_with_llm(self, base_64_image: str, prompt: str) -> AIMessage:
        """
        Process a base64-encoded image with a language model using the provided prompt.
        
        Args:
            base_64_image: Base64 encoded image string
            prompt: Text prompt to send with the image
            
        Returns:
            Language model response
            
        Raises:
            Exception: If there's an error during processing
        """
        try:
            self.prompt_chat =  ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base_64_image}"
                            },
                        },
                    ],
                )
            ])
            parse_doc_chain = self.prompt_chat | self.chat_model | StrOutputParser()
            response = parse_doc_chain.invoke(
                {"prompt": prompt}
            )
            print(response)
            logger.info(f"chat model response: {str(response)}")
            return response
        except Exception as e:
            logger.error(f"Failed to parse document with llm: {str(e)}")
            raise
