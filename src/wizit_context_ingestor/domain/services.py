import base64
import logging
import io
import fitz
from PIL import Image
from typing import List
from ..domain.models import ParsedDocPage, ParsedDoc

logger = logging.getLogger(__name__)

# CHECK THIS THING IMPROVE THE WAY CODE IS STRUCTURED
class ParseDocModelService():
    """
    Class for parsing PDF documents, converting pages to base64 images
    """

    def __init__(self, file_path: str):
        """
        Initialize a PDF document parser.
        
        Args:
            file_path: Path to the PDF file to parse
        """
        self.file_path = file_path
        self.pdf_document = fitz.open(file_path)
        self.page_count = self.pdf_document.page_count

    
    def pdf_page_to_base64(self, page_number: int) -> ParsedDocPage:
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
            b64_encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            logger.info(f"Page {page_number} encoded successfully")
            return ParsedDocPage(
                page_number=page_number, 
                page_base64=b64_encoded_image
            )
        except Exception as e:
            logger.error(f"Failed to parse b64 image: {str(e)}")
            raise

    def parse_document_to_base64(self) -> List[ParsedDocPage]:
        """
        Convert all pages in the PDF document to base64-encoded images.
        
        Returns:
            List of base64 encoded strings for each page
            
        Raises:
            Exception: If there's an error during conversion
        """
        # BASE DE DATOS SINTETICOS DE PREGUNTAS Y RESPUESTAS SOBRE EL DOCUMENTO, FINE TUNING PARA EL LLM
        # GEMMA 2 --> DATASET DE PREGUNTAS Y RESPUESTAS SOBRE EL DOCUMENTO 
        # RAG --> FINETUNING AUTOMATICO / CONSULTAR EL MODELO
        # OPENAI --> PREGUNTAS Y RESPUESTAS SOBRE EL DOCUMENTO
        # COLAB --> PREGUNTAS Y RESPUESTAS SOBRE EL DOCUMENTO
        try:
            base64_pages = []
            for page_number in range(1, self.pdf_document.page_count + 1):
                page = self.pdf_page_to_base64(page_number)
                base64_pages.append(page)
            # logger.info(f"{len(base64_pages)} Pages encoded to base64 successfully")
            return base64_pages
        except Exception as e:
            logger.error(f"Failed to parse b64 image: {str(e)}")
            raise

    def create_md_content(self, parsed_pages: List[ParsedDocPage]) -> ParsedDoc:
        """
        Create a markdown content from a list of parsed pages.
        """
        md_content = ""
        for page in parsed_pages:
            md_content += f"## Page {page.page_number}\n\n"
            md_content += f"{page.page_text}\n\n"
        return ParsedDoc(
            pages=parsed_pages,
            document_text=md_content
        )

    # def 