"""
Domain models for app
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ParsedDocPage:
    """Represents a parsed document page."""
    page_number: int
    page_base64: str
    page_text: Optional[str] = None  

@dataclass
class ParsedDoc:
    """Represents a parsed document."""
    pages: List[ParsedDocPage]
    document_text: str
