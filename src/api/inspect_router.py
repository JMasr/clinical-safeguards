from __future__ import annotations

"""
/v1/inspect — pipeline execution trace endpoint.

Only registered when SAFEGUARD_INSPECT_MODE=true.
Never enable in production deployments that handle real patient data.

Translates the internal PipelineTrace (core/pipeline.py dataclasses) into
the public InspectResponse (models/response.py Pydantic models).
This is the only place where the two representations meet — keeping
the pipeline free to evolve without breaking external consumers.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.core import PipelineTrace, StageTrace
from src.models import InspectResponse, PromptInput, StageTraceResponse

logger = logging.getLogger(__name__)

inspect_router = APIRouter()


def _trace_to_response(trace: PipelineTrace) -> InspectResponse:
    """
    Adapter: PipelineTrace (internal dataclass) → InspectResponse (Pydantic).

    This is the single translation point. If PipelineTrace gains new fields
    (e.g. parallel_group for concurrent stages), this function absorbs the
    change — external consumers stay stable.
    """
    stage_responses = [_stage_trace_to_response(st) for st in trace.stage_traces]
    return InspectResponse(
        final=trace.final_response,
        trace=stage_responses,
        total_duration_ms=trace.total_duration_ms,
    )


def _stage_trace_to_response(st: StageTrace) -> StageTraceResponse:
    return StageTraceResponse(
        stage=st.result.stage_name,
        label=st.result.label,
        score=st.result.confidence,
        triggered_by=st.result.triggered_by,
        short_circuit=st.result.short_circuit,
        duration_ms=st.duration_ms,
    )


@inspect_router.post(
    "/v1/inspect",
    response_model=InspectResponse,
    summary="Evaluate a prompt and return the full pipeline execution trace",
    description=(
            "Returns the same result as /v1/evaluate plus a per-stage breakdown "
            "with label, confidence score, and execution time. "
            "Stages skipped due to short-circuit are absent from `trace`. "
            "Only available when SAFEGUARD_INSPECT_MODE=true. "
            "Do not enable in production deployments."
    ),
)
async def inspect(request: Request, body: PromptInput) -> JSONResponse:
    pipeline = request.app.state.safeguard_pipeline
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    logger.info(
        "Inspect request — correlation_id=%s length=%d",
        correlation_id,
        len(body.text),
    )

    trace: PipelineTrace = pipeline.evaluate_with_trace(body)
    response = _trace_to_response(trace)

    logger.info(
        "Inspect complete — correlation_id=%s code=%s stages_run=%d total_ms=%.1f",
        correlation_id,
        response.final.code,
        len(response.trace),
        response.total_duration_ms,
    )

    return JSONResponse(
        status_code=200,
        content=response.model_dump(mode="json"),
    )
