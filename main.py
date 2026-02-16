"""
Entrypoint for the Clinical Safeguard Middleware.

Hydra manages config composition; uvicorn serves the FastAPI app.

Development (deterministic only, no models):
    python -m clinical_safeguard.main pipeline=default

Production (clinical BERT):
    python -m clinical_safeguard.main pipeline=clinical

Full stack (BERT + attack detection):
    python -m clinical_safeguard.main pipeline=full

CLI overrides (Hydra syntax):
    python -m clinical_safeguard.main pipeline=full pipeline.stages.1.threshold=0.9

Uvicorn direct (uses conf/config.yaml default):
    uvicorn clinical_safeguard.main:app --reload
    → This path skips Hydra — pipeline_cfg comes from _load_default_cfg().
      Use only for quick local testing; prefer python -m ... for full control.
"""
from __future__ import annotations

import hydra
import uvicorn
from omegaconf import DictConfig

from src.api.app import create_app


def _load_default_cfg() -> DictConfig:
    """
    Load the default Hydra config without the @hydra.main decorator.
    Used when the module is imported directly (e.g. `uvicorn main:app`).
    The pipeline group default is resolved from conf/config.yaml.
    """
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    conf_dir = str(Path(__file__).parent / "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        cfg = compose(config_name="config")
    return cfg


# ---------------------------------------------------------------------------
# `uvicorn clinical_safeguard.main:app` path
# Hydra is not involved — config loaded from conf/ directly.
# ---------------------------------------------------------------------------
app = create_app(pipeline_cfg=_load_default_cfg())


# ---------------------------------------------------------------------------
# `python -m clinical_safeguard.main` path — full Hydra CLI
# ---------------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    """
    Entry point when run via `python -m clinical_safeguard.main`.
    Hydra composes the config (with CLI overrides) and passes it here.
    """
    pipeline_app = create_app(pipeline_cfg=cfg)
    uvicorn.run(
        pipeline_app,
        host=cfg.app.get("host", "0.0.0.0"),
        port=cfg.app.get("port", 8000),
        log_level=cfg.app.get("log_level", "info"),
    )


if __name__ == "__main__":  # pragma: no cover
    _hydra_main()
