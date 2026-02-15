from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request

from src.config import initialize_hf_services, Settings, get_settings
from src.core import SafeguardPipeline
from src.core.exceptions import ResourceLoadError
from src.stages.deterministic import DeterministicStage
from src.stages.semantic import SemanticBERTStage

logger = logging.getLogger(__name__)

# Key used to store the pipeline singleton in app.state
_PIPELINE_KEY = "safeguard_pipeline"


def _configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )


def _build_pipeline(settings: Settings) -> SafeguardPipeline:
    """
    Construct the pipeline from settings.
    Called once at startup — ResourceLoadError here is fatal (fast-fail).
    HF services must be initialized before this is called.
    """
    det_stage = DeterministicStage(
        keywords_crisis_path=settings.keywords_crisis_path,
        keywords_malign_path=settings.keywords_malign_path,
        bypass_patterns_path=settings.bypass_patterns_path,
    )

    stages = [det_stage]

    if settings.enable_semantic_stage:
        bert_stage = SemanticBERTStage(
            model_id=settings.model_id,
            threshold=settings.model_threshold,
            inference_timeout_s=settings.inference_timeout_s,
        )
        stages.append(bert_stage)
        logger.info(
            "SemanticBERTStage enabled — model=%s", settings.model_id
        )
    else:
        logger.info("SemanticBERTStage disabled via SAFEGUARD_ENABLE_BERT=false")

    return SafeguardPipeline(stages=stages)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup order:
      1. Configure logging
      2. Load and validate settings (fails fast if HF_TOKEN missing + BERT on)
      3. Authenticate with HuggingFace Hub (only if BERT enabled)
      4. Build the pipeline

    ResourceLoadError at any step → log + re-raise → service does NOT start.
    """
    _configure_logging()
    settings = get_settings()

    logger.info("Clinical Safeguard Middleware starting…")
    try:
        if settings.enable_semantic_stage:
            # hf_token presence is guaranteed by the Settings validator
            initialize_hf_services(settings.hf_token)

        pipeline = _build_pipeline(settings)
        app.state.safeguard_pipeline = pipeline
        logger.info("Pipeline ready. Service is accepting requests.")
    except ResourceLoadError as exc:
        logger.critical(
            "FATAL: Could not initialize safety pipeline: %s. "
            "Service will not start.",
            exc,
        )
        raise

    yield  # service runs here

    logger.info("Clinical Safeguard Middleware shutting down.")


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    App factory. Accepts an optional Settings override for testing.
    In production, get_settings() is used via the lifespan handler.
    """
    if settings is not None:
        # Override the cached singleton for this app instance
        get_settings.cache_clear()

    app = FastAPI(
        title="Clinical Safeguard Middleware",
        version="0.1.0",
        description=(
            "Deterministic + semantic guardrail pipeline for clinical AI assistants. "
            "Returns a standardised JSON response for every prompt evaluated."
        ),
        lifespan=_lifespan,
    )

    # Correlation-ID middleware — every request gets a traceable ID
    @app.middleware("http")
    async def add_correlation_id(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    from src.api.router import router  # noqa: PLC0415
    app.include_router(router)

    return app
