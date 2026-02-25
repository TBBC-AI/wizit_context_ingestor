from logging import getLogger

from langgraph.graph import END, START, StateGraph

from .transcription_nodes import TranscriptionNodes
from .transcription_state import TranscriptionInputState, TranscriptionState

# from .transcription_tools import transcribe_page, correct_transcription

logger = getLogger(__name__)


class TranscriptionWorkflow:
    __slots__ = (
        "llm_model",
        "transcription_nodes",
        "transcription_additional_instructions",
    )

    def __init__(self, llm_model, transcription_additional_instructions):
        self.llm_model = llm_model
        self.transcription_additional_instructions = (
            transcription_additional_instructions
        )
        self.transcription_nodes = TranscriptionNodes(
            self.llm_model, self.transcription_additional_instructions
        )

    def gen_workflow(self) -> StateGraph:
        try:
            workflow = StateGraph(
                TranscriptionState, input_schema=TranscriptionInputState
            )
            workflow.add_node("transcribe", self.transcription_nodes.transcribe)
            workflow.add_node(
                "check_transcription", self.transcription_nodes.check_transcription
            )
            workflow.add_node(
                "validate_transcription_results",
                self.transcription_nodes.validate_transcription_results,
            )
            workflow.add_edge(START, "transcribe")
            # workflow.add_edge("transcribe", "validate_transcription_results")
            return workflow
        except Exception as e:
            logger.error(f"Error generating transcription workflow: {e}")
            raise e
