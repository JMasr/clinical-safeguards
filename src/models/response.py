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
    # Always True for Crisis and Malign — no configurable override.
    short_circuit: bool = False


# ---------------------------------------------------------------------------
# Final HTTP response payload
# ---------------------------------------------------------------------------

class ResponseData(BaseModel):
    processed_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: dict


class FinalResponse(BaseModel):
    code: ResponseCode
    label: Label
    data: ResponseData


# ---------------------------------------------------------------------------
# Inspect endpoint response — HTTP adapter over PipelineTrace
#
# These models are the public contract of /v1/inspect. They are deliberately
# separate from StageTrace/PipelineTrace (internal dataclasses in core/pipeline.py)
# so the pipeline can evolve its internal representation without breaking
# external consumers. The translation happens in api/inspect_router.py.
# ---------------------------------------------------------------------------

class StageTraceResponse(BaseModel):
    """Public representation of a single stage execution record."""

    stage: str
    label: Label
    score: float = Field(ge=0.0, le=1.0)
    triggered_by: str | None = None
    short_circuit: bool
    duration_ms: float = Field(ge=0.0)


class InspectResponse(BaseModel):
    """
    Full pipeline execution trace returned by /v1/inspect.

    `trace` contains only the stages that actually ran.
    `skipped_stages` lists stages that were registered but not executed
    due to a short-circuit — in execution order. Together, trace +
    skipped_stages always reconstruct the full pipeline without needing
    a separate call to /health.
    """

    final: FinalResponse
    trace: list[StageTraceResponse]
    skipped_stages: list[str]
    total_stages: int = Field(ge=0)
    total_duration_ms: float = Field(ge=0.0)