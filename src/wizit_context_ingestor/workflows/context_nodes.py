from ..data.prompts import WORKFLOW_CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.pregel.main import Command
from .context_state import ContextState


class ContextNodes:
    def __init__(self, llm_model, tools, context_additional_instructions):
        self.llm_model = llm_model
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.context_additional_instructions = context_additional_instructions

    def gen_context(self, state: ContextState, config):
        try:
            messages = state["messages"]
            document_content = state["document_content"]
            if not messages:
                raise ValueError("No messages provided")
            # parser = PydanticOutputParser(pydantic_object=Transcription)
            # format_instructions=parser.get_format_instructions(),
            formatted_context_system_prompt = WORKFLOW_CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT.format(
                context_additional_instructions=self.context_additional_instructions,
                document_content=document_content,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=formatted_context_system_prompt),
                    MessagesPlaceholder("messages"),
                ]
            )
            model_with_structured_output = self.llm_model.bind_tools(self.tools)
            context_chain = prompt | model_with_structured_output
            context_result = context_chain.invoke({"messages": messages})
            return {"messages": [context_result]}
        except Exception as e:
            print(f"Error occurred: {e}")
            raise e

    def return_context(self, state: ContextState, config):
        latest_message = state["messages"][-1]
        if type(latest_message) is ToolMessage:
            return Command(goto=END, update={"context": latest_message.content})
        else:
            raise ValueError("Invalid message type to return context")

    def tool_node(self, state: ContextState, config):
        messages = state["messages"]
        tool_calls = messages[-1].tool_calls
        should_end_workflow = False
        observations = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool = self.tools_by_name[tool_name]
            tool_result = tool.invoke(tool_call["args"])
            observations.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
            if tool_call["name"] == "complete_context_gen":
                should_end_workflow = True

        if should_end_workflow:
            return Command(goto="return_context", update={"messages": observations})
        else:
            return Command(goto="gen_context", update={"messages": observations})
