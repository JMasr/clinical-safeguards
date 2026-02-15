from pydantic import BaseModel, Field, field_validator


class PromptInput(BaseModel):
    model_config = {"frozen": True}  # Immutability

    text: str = Field(..., min_length=1, max_length=8192)
    session_id: str | None = Field(default=None)

    @field_validator("text")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text cannot be empty or whitespace")
        return stripped
