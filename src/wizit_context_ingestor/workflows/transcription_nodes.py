from ..data.prompts import (
    AGENT_TRANSCRIPTION_SYSTEM_PROMPT,
    IMAGE_TRANSCRIPTION_CHECK_SYSTEM_PROMPT,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.pregel.main import Command
from .transcription_schemas import Transcription, TranscriptionCheck
from .transcription_state import TranscriptionState


class TranscriptionNodes:
    __slots__ = ("llm_model", "transcription_additional_instructions")

    def __init__(self, llm_model, transcription_additional_instructions):
        self.llm_model = llm_model
        self.transcription_additional_instructions = (
            transcription_additional_instructions
        )

    def transcribe(self, state: TranscriptionState, config):
        try:
            messages = state["messages"]
            transcription_notes = ""
            if "transcription_notes" in state.keys():
                transcription_notes = state["transcription_notes"]
            if not messages:
                raise ValueError("No messages provided")
            # parser = PydanticOutputParser(pydantic_object=Transcription)
            # format_instructions=parser.get_format_instructions(),
            formatted_transcription_system_prompt = AGENT_TRANSCRIPTION_SYSTEM_PROMPT.format(
                transcription_additional_instructions=self.transcription_additional_instructions,
                transcription_notes=transcription_notes,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=formatted_transcription_system_prompt),
                    MessagesPlaceholder("messages"),
                ]
            )
            model_with_structured_output = self.llm_model.with_structured_output(
                Transcription
            )
            transcription_chain = prompt | model_with_structured_output
            transcription_result = transcription_chain.invoke({"messages": messages})
            return Command(
                goto="check_transcription",
                update={
                    "transcription": transcription_result.transcription,
                    "transcription_status": "in_progress",
                },
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
            # parser = PydanticOutputParser(pydantic_object=TranscriptionCheck)

            formatted_image_transcription_check_system_prompt = IMAGE_TRANSCRIPTION_CHECK_SYSTEM_PROMPT.format(
                transcription_additional_instructions=self.transcription_additional_instructions,
                transcription=transcription,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=formatted_image_transcription_check_system_prompt
                    ),
                    MessagesPlaceholder("messages"),
                ]
            )
            model_with_structured_output = self.llm_model.with_structured_output(
                TranscriptionCheck
            )
            transcription_check_chain = prompt | model_with_structured_output
            transcription_check_result = transcription_check_chain.invoke(
                {"transcription": transcription, "messages": messages}
            )
            return Command(
                goto="validate_transcription_results",
                update={
                    "transcription_accuracy": transcription_check_result.transcription_accuracy,
                    "transcription_notes": transcription_check_result.transcription_notes,
                },
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            return Command(goto=END, update={"transcription_accuracy": 0.0})

    def validate_transcription_results(self, state, config):
        try:
            if "transcription_accuracy" not in state:
                raise ValueError("Missing 'transcription_accuracy' in state")

            if "transcription_retries" not in state:
                transcription_retries = 0
            else:
                transcription_retries = state["transcription_retries"]

            transcription_accuracy = state["transcription_accuracy"]

            max_transcription_retries = config["configurable"][
                "max_transcription_retries"
            ]
            transcription_accuracy_threshold = config["configurable"][
                "transcription_accuracy_threshold"
            ]

            if transcription_accuracy < transcription_accuracy_threshold:
                if transcription_retries < max_transcription_retries:
                    # retry transcription
                    return Command(
                        goto="transcribe",
                        update={
                            "transcription_retries": transcription_retries + 1,
                            "transcription_accuracy": 0.0,
                            "transcription_status": "failed",
                        },
                    )
                else:
                    return Command(goto=END, update={"transcription_status": "failed"})
            else:
                # success
                return Command(goto=END, update={"transcription_status": "completed"})
        except Exception as e:
            print(f"Error occurred: {e}")
            return Command(goto=END, update={"transcription_status": "failed"})
