from __future__ import annotations

"""
Unit tests for api/app.py.

Pipeline config is now a Hydra DictConfig, not a Settings object.
Tests build DictConfig directly with OmegaConf.create() — no file I/O,
no Hydra initialization overhead, no model downloads.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf, DictConfig

from src.api.app import (
    _build_pipeline,
    _configure_logging,
    _needs_hf_auth,
    create_app,
)
from src.config import RESOURCES
from src.core.exceptions import ResourceLoadError
from tests.conftest import make_settings

DET_TARGET = "src.stages.deterministic.DeterministicStage"
BERT_TARGET = "src.stages.semantic.SemanticBERTStage"
ATTACK_TARGET = "src.stages.attack_detection.AttackDetectionStage"
UNKNOWN_TARGET = "src.stages.invented.FakeStage"


def _det_cfg() -> dict:
    return {
        "_target_": DET_TARGET,
        "keywords_crisis_path": str(RESOURCES / "keywords_crisis.yaml"),
        "keywords_malign_path": str(RESOURCES / "keywords_malign.yaml"),
        "bypass_patterns_path": str(RESOURCES / "bypass_patterns.yaml"),
    }


def _bert_cfg(threshold: float = 0.75) -> dict:
    return {
        "_target_": BERT_TARGET,
        "model_id": "test/model",
        "threshold": threshold,
        "inference_timeout_s": 5,
    }


def _attack_cfg() -> dict:
    return {
        "_target_": ATTACK_TARGET,
        "model_id": "test/attack-model",
        "threshold": 0.85,
        "inference_timeout_s": 5,
    }


def _stages_cfg(*stage_dicts: dict) -> DictConfig:
    return OmegaConf.create({"stages": list(stage_dicts)})


def _pipeline_cfg(*stage_dicts: dict) -> DictConfig:
    pipeline = OmegaConf.create({"pipeline": _stages_cfg(*stage_dicts)})
    return pipeline

# ---------------------------------------------------------------------------
# _configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_configure_logging_does_not_raise(self) -> None:
        _configure_logging()


# ---------------------------------------------------------------------------
# _needs_hf_auth
# ---------------------------------------------------------------------------

class TestNeedsHfAuth:
    def test_deterministic_only_does_not_need_hf(self) -> None:
        cfg = _stages_cfg(_det_cfg())
        assert _needs_hf_auth(cfg) is False

    def test_bert_stage_needs_hf(self) -> None:
        cfg = _stages_cfg(_det_cfg(), _bert_cfg())
        assert _needs_hf_auth(cfg) is True

    def test_attack_stage_needs_hf(self) -> None:
        cfg = _stages_cfg(_det_cfg(), _attack_cfg())
        assert _needs_hf_auth(cfg) is True


# ---------------------------------------------------------------------------
# _build_pipeline — registry validation
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_deterministic_only_pipeline(self) -> None:
        cfg = _stages_cfg(_det_cfg())
        pipeline = _build_pipeline(cfg)
        assert len(pipeline._stages) == 1
        assert pipeline._stages[0].name == "deterministic"

    def test_two_stage_pipeline_order_preserved(self) -> None:
        cfg = _stages_cfg(_det_cfg(), _bert_cfg())
        pipeline = _build_pipeline(cfg)
        assert len(pipeline._stages) == 2
        assert pipeline._stages[0].name == "deterministic"
        assert pipeline._stages[1].name == "semantic_bert"

    def test_three_stage_pipeline(self) -> None:
        cfg = _stages_cfg(_det_cfg(), _bert_cfg(), _attack_cfg())
        pipeline = _build_pipeline(cfg)
        assert len(pipeline._stages) == 3
        assert pipeline._stages[2].name == "attack_detection"

    def test_attack_before_bert_order_respected(self) -> None:
        """Hydra config order is the execution order — no hardcoded ordering."""
        cfg = _stages_cfg(_det_cfg(), _attack_cfg(), _bert_cfg())
        pipeline = _build_pipeline(cfg)
        assert pipeline._stages[1].name == "attack_detection"
        assert pipeline._stages[2].name == "semantic_bert"

    def test_unknown_target_raises_resource_load_error(self) -> None:
        cfg = _stages_cfg(_det_cfg(), {"_target_": UNKNOWN_TARGET})
        with pytest.raises(ResourceLoadError, match="Unknown stage"):
            _build_pipeline(cfg)

    def test_resource_load_error_propagates_on_bad_paths(self) -> None:
        cfg = _stages_cfg({
            "_target_": DET_TARGET,
            "keywords_crisis_path": "/nonexistent/crisis.yaml",
            "keywords_malign_path": "/nonexistent/malign.yaml",
            "bypass_patterns_path": "/nonexistent/bypass.yaml",
        })
        with pytest.raises(ResourceLoadError):
            _build_pipeline(cfg)

    def test_threshold_override_respected(self) -> None:
        cfg = _stages_cfg(_det_cfg(), _bert_cfg(threshold=0.99))
        pipeline = _build_pipeline(cfg)
        assert pipeline._stages[1]._threshold == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------

class TestCreateApp:
    def test_create_app_returns_fastapi_instance(self) -> None:
        from fastapi import FastAPI  # noqa: PLC0415
        cfg = _pipeline_cfg(_det_cfg())
        app = create_app(pipeline_cfg=cfg)
        assert isinstance(app, FastAPI)

    def test_create_app_includes_evaluate_route(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(_det_cfg()))
        routes = [r.path for r in app.routes]
        assert "/v1/evaluate" in routes

    def test_create_app_includes_health_route(self) -> None:
        app = create_app(pipeline_cfg=_pipeline_cfg(_det_cfg()))
        routes = [r.path for r in app.routes]
        assert "/health" in routes

    def test_create_app_stores_pipeline_cfg_on_state(self) -> None:
        cfg = _pipeline_cfg(_det_cfg())
        app = create_app(pipeline_cfg=cfg)
        assert app.state.pipeline_cfg is cfg


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

class TestLifespan:
    def test_startup_loads_pipeline_into_app_state(self) -> None:
        pipeline_cfg = _pipeline_cfg(_det_cfg())
        app = create_app(pipeline_cfg=pipeline_cfg)
        with TestClient(app) as client:
            assert hasattr(app.state, "safeguard_pipeline")
            assert client.get("/health").status_code == 200

    def test_startup_with_hf_stage_calls_hf_init(self) -> None:
        pipeline_cfg = _pipeline_cfg(_det_cfg(), _bert_cfg())
        settings = make_settings(HF_TOKEN="hf_testtoken")
        app = create_app(pipeline_cfg=pipeline_cfg, settings=settings)

        with patch("src.api.app.get_settings", return_value=settings), \
                patch("src.api.app.initialize_hf_services") as mock_init:
            with TestClient(app):
                mock_init.assert_called_once_with(settings.hf_token)

    def test_startup_deterministic_only_skips_hf_init(self) -> None:
        pipeline_cfg = _pipeline_cfg(_det_cfg())
        settings = make_settings()
        app = create_app(pipeline_cfg=pipeline_cfg, settings=settings)

        with patch("src.api.app.get_settings", return_value=settings), \
                patch("src.api.app.initialize_hf_services") as mock_init:
            with TestClient(app):
                mock_init.assert_not_called()

    def test_startup_hf_stage_without_token_raises(self) -> None:
        cfg = _pipeline_cfg(_det_cfg(), _bert_cfg())
        settings = make_settings(HF_TOKEN=None)
        app = create_app(pipeline_cfg=cfg, settings=settings)

        with patch("src.api.app.get_settings", return_value=settings):
            with pytest.raises(Exception):
                with TestClient(app, raise_server_exceptions=True):
                    pass

    def test_startup_failure_bad_paths_raises(self, tmp_path: Path) -> None:
        bad_cfg = _pipeline_cfg({
            "_target_": DET_TARGET,
            "keywords_crisis_path": str(tmp_path / "no.yaml"),
            "keywords_malign_path": str(tmp_path / "no.yaml"),
            "bypass_patterns_path": str(tmp_path / "no.yaml"),
        })
        app = create_app(pipeline_cfg=bad_cfg)
        with pytest.raises(Exception):
            with TestClient(app, raise_server_exceptions=True):
                pass


# ---------------------------------------------------------------------------
# main.py
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_app_is_importable(self) -> None:
        from fastapi import FastAPI  # noqa: PLC0415
        from main import app as main_app  # noqa: PLC0415
        assert isinstance(main_app, FastAPI)

    def test_hydra_main_callable(self) -> None:
        from unittest.mock import patch as _patch  # noqa: PLC0415
        from omegaconf import OmegaConf as _OC  # noqa: PLC0415
        from main import _hydra_main  # noqa: PLC0415

        cfg = _OC.create({
            "app": {"host": "0.0.0.0", "port": 8000, "log_level": "info"},
            "pipeline": {"stages": [{
                "_target_": "src.stages.deterministic.DeterministicStage",
                "keywords_crisis_path": str(RESOURCES / "keywords_crisis.yaml"),
                "keywords_malign_path": str(RESOURCES / "keywords_malign.yaml"),
                "bypass_patterns_path": str(RESOURCES / "bypass_patterns.yaml"),
            }]},
            "paths": {}
        })
        with _patch("main.uvicorn.run") as mock_run:
            _hydra_main.__wrapped__(cfg)
            mock_run.assert_called_once()
