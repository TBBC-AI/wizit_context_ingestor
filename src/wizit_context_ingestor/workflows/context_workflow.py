from logging import getLogger

from langgraph.graph import END, START, StateGraph

from .context_nodes import ContextNodes
from .context_state import ContextState
from .context_tools import complete_context_gen

logger = getLogger(__name__)


class ContextWorkflow:
    __slots__ = (
        "llm_model",
        "tools",
        "context_nodes",
        "context_additional_instructions",
    )

    def __init__(self, llm_model, context_additional_instructions):
        self.llm_model = llm_model
        self.context_additional_instructions = context_additional_instructions
        self.tools = [complete_context_gen]
        self.context_nodes = ContextNodes(
            self.llm_model, self.tools, self.context_additional_instructions
        )

    def gen_workflow(self):
        try:
            workflow = StateGraph(ContextState)
            workflow.add_node("gen_context", self.context_nodes.gen_context)
            workflow.add_node("tools", self.context_nodes.tool_node)
            workflow.add_node("return_context", self.context_nodes.return_context)
            workflow.add_edge(START, "gen_context")
            workflow.add_edge("gen_context", "tools")
            workflow.add_edge("return_context", END)
            return workflow
        except Exception as e:
            logger.error(f"Error generating context workflow: {e}")
            raise e
