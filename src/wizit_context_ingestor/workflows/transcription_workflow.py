from langgraph.graph import StateGraph
from langgraph.graph import START, END
from .transcription_state import TranscriptionState
from .transcription_nodes import TranscriptionNodes
# from .transcription_tools import transcribe_page, correct_transcription


class TranscriptionWorkflow:
    __slots__ = ("llm_model", "transcription_nodes")

    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.transcription_nodes = TranscriptionNodes(self.llm_model)

    def gen_workflow(self):
        workflow = StateGraph(TranscriptionState)
        workflow.add_node("transcribe", self.transcription_nodes.transcribe)
        workflow.add_node(
            "check_transcription", self.transcription_nodes.check_transcription
        )
        workflow.add_edge(START, "transcribe")
        return workflow
