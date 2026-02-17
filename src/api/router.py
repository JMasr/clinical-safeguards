from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.models import FinalResponse, PromptInput

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/evaluate",
    response_model=FinalResponse,
    summary="Evaluate a user prompt through the safeguard pipeline",
    description=(
            "Runs the prompt through the deterministic and (optionally) semantic "
            "stages. Always returns HTTP 200 with a structured JSON body. "
            "The `code` field in the response body carries the business result: "
            "100=Válida, 400=Maligna, 406=Crisis, 500=Server Error."
    ),
)
async def evaluate(request: Request, body: PromptInput) -> JSONResponse:
    """
    HTTP 200 is always returned — the business result lives in `code`.
    This prevents upstream retry logic from re-submitting blocked prompts.
    """
    pipeline = request.app.state.safeguard_pipeline
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    logger.info(
        "Evaluating prompt — correlation_id=%s session_id=%s length=%d",
        correlation_id,
        body.session_id,
        len(body.text),
    )

    response: FinalResponse = pipeline.evaluate(body)

    logger.info(
        "Evaluation complete — correlation_id=%s code=%s etiqueta=%s",
        correlation_id,
        response.code,
        response.etiqueta,
    )

    return JSONResponse(
        status_code=200,
        content=response.model_dump(mode="json"),
    )


@router.get(
    "/health",
    summary="Health check",
    description=(
            "Returns 200 if the service is running and the pipeline is loaded. "
            "The `pipeline` field lists the active stages in execution order "
            "and whether inspect mode is enabled."
    ),
)
async def health(request: Request) -> JSONResponse:
    pipeline = getattr(request.app.state, "safeguard_pipeline", None)
    if pipeline is None:
        return JSONResponse(status_code=503, content={"status": "unavailable"})

    import os  # noqa: PLC0415
    inspect_mode = os.getenv("SAFEGUARD_INSPECT_MODE", "false").lower() == "true"

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "pipeline": {
                "stages": list(pipeline.stage_names),
                "stage_count": len(pipeline.stage_names),
                "inspect_mode": inspect_mode,
            },
        },
    )
