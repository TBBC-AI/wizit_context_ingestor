from ..application.interfaces import PersistenceService
from ..domain.models import ParsedDoc
class LocalPersistenceService(PersistenceService):
    """Local persistence service."""

    def save_parsed_document(self, file_name: str, parsed_document: ParsedDoc):
        """Save a parsed document."""
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(parsed_document.document_text)