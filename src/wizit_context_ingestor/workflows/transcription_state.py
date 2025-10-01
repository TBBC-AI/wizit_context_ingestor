from typing_extensions import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TranscriptionState(TypedDict):
    # id: str
    # source_file_markdown: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    transcription: str
    # status: Literal["pending", "in_progress", "completed", "failed"]
