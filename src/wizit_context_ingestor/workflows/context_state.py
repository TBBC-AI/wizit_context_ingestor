from typing_extensions import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ContextState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_content: str
    context: str
    context_relevance: float
