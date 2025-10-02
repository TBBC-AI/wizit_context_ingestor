from typing import Literal
from typing_extensions import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TranscriptionInputState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class TranscriptionState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    transcription: str
    transcription_retries: int
    transcription_notes: str
    transcription_status: Literal["pending", "in_progress", "completed", "failed"]
    transcription_accuracy: float
