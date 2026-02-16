from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from omegaconf import DictConfig

from src.config import initialize_hf_services, Settings, get_settings
from src.core import SafeguardPipeline, ResourceLoadError
from src.stages import STAGE_REGISTRY

logger = logging.getLogger(__name__)

_PIPELINE_KEY = "safeguard_pipeline"

# Fully-qualified names of stage classes that require a HuggingFace model.
# Used to decide whether HF auth is needed before building the pipeline.
_HF_DEPENDENT_STAGES: frozenset[str] = frozenset(
    {
        "src.stages.semantic.SemanticBERTStage",
        "src.stages.attack_detection.AttackDetectionStage",
    }
)


def _configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )


def _build_pipeline(full_cfg: DictConfig) -> SafeguardPipeline:
    """
    Instantiate a SafeguardPipeline from a Hydra DictConfig.

    Each entry in pipeline_cfg.stages must have a '_target_' key that
    resolves to a class in STAGE_REGISTRY. This is validated before any
    import or instantiation happens — an unrecognised _target_ raises
    ResourceLoadError immediately (fail-fast, not fail-silent).

    Stage order in the pipeline matches the order in the YAML config.
    """
    from hydra.utils import instantiate  # noqa: PLC0415 — lazy import keeps startup fast
    pipeline_cfg = full_cfg.pipeline
    stages_cfg = list(pipeline_cfg.stages)

    # --- Registry validation -------------------------------------------
    unknown = [
        cfg["_target_"]
        for cfg in stages_cfg
        if cfg.get("_target_") not in STAGE_REGISTRY
    ]
    if unknown:
        raise ResourceLoadError(
            f"Unknown stage(s) in pipeline config: {unknown}. "
            f"Valid stages are: {sorted(STAGE_REGISTRY)}"
        )

    # --- Instantiation ------------------------------------------------
    # hydra.utils.instantiate() wraps constructor errors in InstantiationException.
    # We unwrap it here so callers always get ResourceLoadError on failure.
    from hydra.errors import InstantiationException  # noqa: PLC0415

    stages = []
    for stage_cfg in stages_cfg:
        try:
            stage = instantiate(stage_cfg)
        except InstantiationException as exc:
            cause = exc.__cause__ or exc
            raise ResourceLoadError(
                f"Failed to instantiate stage '{stage_cfg.get('_target_')}'", cause
            ) from exc
        logger.info("Stage loaded: %s", stage.name)
        stages.append(stage)

    return SafeguardPipeline(stages=stages)


def _needs_hf_auth(full_cfg: DictConfig) -> bool:
    """Return True if any stage in the config requires a HuggingFace model."""
    return any(
        cfg.get("_target_") in _HF_DEPENDENT_STAGES
        for cfg in full_cfg.pipeline.stages
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup order:
      1. Configure logging
      2. Load settings (secrets + infra config from env/.env)
      3. Read pipeline config (Hydra DictConfig stored on app.state at creation)
      4. Authenticate with HuggingFace Hub if any stage needs it
      5. Build and validate the pipeline

    ResourceLoadError at any step → log + re-raise → service does NOT start.
    """
    _configure_logging()
    settings: Settings = get_settings()
    pipeline_cfg: DictConfig = app.state.pipeline_cfg

    logger.info("Clinical Safeguard Middleware starting…")
    try:
        if _needs_hf_auth(pipeline_cfg):
            if not settings.hf_token:
                raise ResourceLoadError(
                    "HF_TOKEN is required for the configured pipeline stages "
                    "but was not found in environment or .env file."
                )
            initialize_hf_services(settings.hf_token)

        pipeline = _build_pipeline(pipeline_cfg)
        app.state.safeguard_pipeline = pipeline
        logger.info("Pipeline ready with %d stage(s). Accepting requests.", len(pipeline._stages))
    except ResourceLoadError as exc:
        logger.critical(
            "FATAL: Could not initialize safety pipeline: %s. Service will not start.", exc
        )
        raise

    yield

    logger.info("Clinical Safeguard Middleware shutting down.")


def create_app(
        pipeline_cfg: DictConfig | None = None,
        settings: Settings | None = None,
) -> FastAPI:
    """
    App factory.

    Args:
        pipeline_cfg: Hydra DictConfig for the pipeline. In production this
                      comes from @hydra.main(); in tests it is built directly
                      with OmegaConf.create({...}).
        settings:     Optional Settings override. Used in tests to inject a
                      Settings instance without reading .env.
    """
    if settings is not None:
        get_settings.cache_clear()

    app = FastAPI(
        title="Clinical Safeguard Middleware",
        version="0.2.0",
        description=(
            "Deterministic + semantic guardrail pipeline for clinical AI assistants. "
            "Pipeline composition is declared in Hydra YAML configs."
        ),
        lifespan=_lifespan,
    )

    # Store pipeline_cfg on app.state so _lifespan can read it.
    # OmegaConf.create({}) gives an empty fallback — lifespan will raise
    # ResourceLoadError if stages list is empty.
    from omegaconf import OmegaConf  # noqa: PLC0415

    app.state.pipeline_cfg = pipeline_cfg if pipeline_cfg is not None else OmegaConf.create(
        {"stages": []}
    )

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