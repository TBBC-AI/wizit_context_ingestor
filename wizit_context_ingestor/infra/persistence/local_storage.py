from ...application.interfaces import PersistenceService
from ...domain.models import ParsedDoc
import logging
import os
logger = logging.getLogger(__name__)

class LocalStorageService(PersistenceService):
    """Persistence service for local storage."""

    def __init__(self):
        self.tmp_folder = "tmp"

    def load_markdown_file_content(self, file_key: str):
        """Load markdown file content from local storage."""
        file_content = None
        with open(f"{self.tmp_folder}/{file_key}", "r", encoding="utf-8") as file:
            file_content = file.read()
        return file_content


    def retrieve_raw_file(self, file_key: str):
        """Retrieve file path in tmp folder from local storage.
        
        Args:
            file_key: The key (path) of the file in local storage
            
        Returns:
            str: The path of the file in tmp folder
            
        Raises:
            ClientError: If there's an error retrieving the object from local storage
        """
        try:
            tmp_file_path = f"{self.tmp_folder}/{file_key}"
            if not os.path.exists(tmp_file_path):
                raise FileNotFoundError(f"File {file_key} not found in local storage")
            return tmp_file_path
        except Exception as e:
            logger.error(f"Unexpected error retrieving file {file_key} from local storage: {str(e)}")
            raise


    def save_parsed_document(self, file_key: str, parsed_document: ParsedDoc):
        """Save a parsed document."""
        with open(f"{self.tmp_folder}/{file_key}", "w", encoding="utf-8") as f:
            f.write(parsed_document.document_text)