from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated


@tool(parse_docstring=True)
def transcribe_page(image_base_64: Annotated[str, InjectedToolArg]) -> str:
    """Transcribe a document using the provided text.

    Args:
        image_base_64: Base64 encoded image string containing the document to transcribe.

    Returns:
        The transcribed text content from the document.
    """


@tool(parse_docstring=True)
def correct_transcription(
    transcription: str, image_base_64: Annotated[str, InjectedToolArg]
) -> [str, bool]:
    """Correct a transcription using the provided text.

    Args:
        transcription: The transcribed content.
        image_base_64: Base64 encoded image string containing the document to transcribe.

    Returns:
       The corrected transcription.
       The transcription has been executed successfully.
    """


@tool(parse_docstring=True)
def think_tool(reasoning: str) -> str:
    """Reason about the current task and next steps.
    Args:
        reasoning: The reasoning content.

    Returns:
       The reasoning content.
    """
    pass


@tool(parse_docstring=True)
def finish(transcription: str) -> str:
    """Execute a transcription using the provided text.
    Args:
        transcription: The transcribed content.

    Returns:
       The executed transcription.
    """
    pass
