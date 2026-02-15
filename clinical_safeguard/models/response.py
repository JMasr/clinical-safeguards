from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .enums import Label, ResponseCode


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PromptInput(BaseModel):
    """Immutable input contract. Never mutated by any pipeline stage."""

    model_config = {"frozen": True}

    text: str = Field(..., min_length=1, max_length=8192)
    session_id: str | None = Field(default=None)

    @field_validator("text")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text cannot be empty or whitespace only")
        return stripped


# ---------------------------------------------------------------------------
# Intra-pipeline result (produced by each stage)
# ---------------------------------------------------------------------------

class StageResult(BaseModel):
    """Immutable result produced by a single GuardrailStage."""

    model_config = {"frozen": True}

    stage_name: str
    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    # Human-readable reason: keyword matched, pattern id, model score, etc.
    triggered_by: str | None = None
    # When True the pipeline stops immediately after this stage.
    # Always True for Crisis and Maligna — no configurable override.
    short_circuit: bool = False


# ---------------------------------------------------------------------------
# Final HTTP response payload
# ---------------------------------------------------------------------------

class ResponseData(BaseModel):
    texto_procesado: str
    score_confianza: float = Field(ge=0.0, le=1.0)
    metadatos: dict


class FinalResponse(BaseModel):
    code: ResponseCode
    etiqueta: Label
    data: ResponseData
