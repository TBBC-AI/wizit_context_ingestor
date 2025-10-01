from ..data.prompts import (
    IMAGE_TRANSCRIPTION_SYSTEM_PROMPT,
    IMAGE_TRANSCRIPTION_CHECK_SYSTEM_PROMPT,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.pregel.main import Command
from .transcription_schemas import Transcription, TranscriptionCheck


class TranscriptionNodes:
    __slots__ = ("llm_model", "transcription_additional_instructions")

    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.transcription_additional_instructions = "Transcribe the image carefully"

    def transcribe(self, state, config):
        try:
            messages = state["messages"]
            if not messages:
                raise ValueError("No messages provided")
            parser = PydanticOutputParser(pydantic_object=Transcription)
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=IMAGE_TRANSCRIPTION_SYSTEM_PROMPT),
                    MessagesPlaceholder("messages"),
                ]
            ).partial(
                transcription_additional_instructions=self.transcription_additional_instructions,
                format_instructions=parser.get_format_instructions(),
            )
            model_with_structured_output = self.llm_model.with_structured_output(
                Transcription
            )
            transcription_chain = prompt | model_with_structured_output
            transcription_result = transcription_chain.invoke({"messages": messages})
            breakpoint()
            return Command(
                goto="check_transcription",
                update={"transcription": transcription_result.transcription},
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            return Command(goto=END)

    def check_transcription(self, state, config):
        try:
            transcription = state["transcription"]
            messages = state["messages"]
            print("last message, ", messages[-1])
            if not transcription:
                raise ValueError("No transcription provided")
            parser = PydanticOutputParser(pydantic_object=TranscriptionCheck)
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=IMAGE_TRANSCRIPTION_CHECK_SYSTEM_PROMPT),
                    MessagesPlaceholder("messages"),
                ]
            ).partial(
                format_instructions=parser.get_format_instructions(),
            )
            model_with_structured_output = self.llm_model.with_structured_output(
                TranscriptionCheck
            )
            transcription_check_chain = prompt | model_with_structured_output
            transcription_check_result = transcription_check_chain.invoke(
                {"transcription": transcription, "messages": messages}
            )
            breakpoint()
            return Command(goto=END, update={"transcription": "my transcription"})
        except Exception as e:
            print(f"Error occurred: {e}")
            return Command(goto=END)
