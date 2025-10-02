from pydantic import BaseModel, Field


class Transcription(BaseModel):
    """Schema for transcription."""

    transcription: str = Field(
        description="document page transcription",
    )


class TranscriptionCheck(BaseModel):
    """Schema for transcription check."""

    is_correct_transcription: bool = Field(
        description="is a correct transcription",
    )

    transcription_accuracy: float = Field(
        description="transcription accuracy from 0.0 to 1.0",
    )

    transcription_notes: str = Field(
        description="why is a correct transcription or not, why transcription accuracy is not 100%",
    )
