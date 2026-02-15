from __future__ import annotations

"""
Unit tests for api/app.py — covers _build_pipeline branches, lifespan,
create_app with settings override, and the logging setup.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from clinical_safeguard.api.app import _build_pipeline, _configure_logging, create_app
from clinical_safeguard.config.settings import Settings, get_settings, RESOURCES
from clinical_safeguard.core.exceptions import ResourceLoadError
from tests.conftest import make_settings


def _test_settings(**overrides) -> Settings:
    """Build a Settings instance pointed at real resource files, .env isolated."""
    enable_bert = overrides.get("enable_bert", False)
    kwargs = {
        "SAFEGUARD_CRISIS_KEYWORDS": str(RESOURCES / "keywords_crisis.yaml"),
        "SAFEGUARD_MALIGN_KEYWORDS": str(RESOURCES / "keywords_malign.yaml"),
        "SAFEGUARD_BYPASS_PATTERNS": str(RESOURCES / "bypass_patterns.yaml"),
        "SAFEGUARD_ENABLE_BERT": str(enable_bert),
        "SAFEGUARD_MODEL_ID": overrides.get("model_id", "test/model"),
    }
    if enable_bert:
        kwargs["HF_TOKEN"] = overrides.get("hf_token", "hf_testtoken123")
    return make_settings(**kwargs)


# ---------------------------------------------------------------------------
# _configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_configure_logging_does_not_raise(self) -> None:
        _configure_logging()  # must be idempotent


# ---------------------------------------------------------------------------
# _build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_build_pipeline_with_bert_disabled(self) -> None:
        settings = _test_settings(enable_bert=False)
        pipeline = _build_pipeline(settings)
        assert len(pipeline._stages) == 1
        assert pipeline._stages[0].name == "deterministic"

    def test_build_pipeline_with_bert_enabled(self) -> None:
        settings = _test_settings(enable_bert=True)
        pipeline = _build_pipeline(settings)
        assert len(pipeline._stages) == 2
        assert pipeline._stages[0].name == "deterministic"
        assert pipeline._stages[1].name == "semantic_bert"

    def test_build_pipeline_resource_load_error_propagates(
            self, tmp_path: Path
    ) -> None:
        settings = make_settings(
            SAFEGUARD_CRISIS_KEYWORDS=str(tmp_path / "nonexistent.yaml"),
            SAFEGUARD_MALIGN_KEYWORDS=str(tmp_path / "nonexistent.yaml"),
            SAFEGUARD_BYPASS_PATTERNS=str(tmp_path / "nonexistent.yaml"),
            SAFEGUARD_ENABLE_BERT="false",
        )
        with pytest.raises(ResourceLoadError):
            _build_pipeline(settings)


# ---------------------------------------------------------------------------
# create_app — settings injection
# ---------------------------------------------------------------------------

class TestCreateApp:
    def test_create_app_with_settings_clears_cache(self) -> None:
        settings = _test_settings()
        get_settings.cache_clear()
        app = create_app(settings=settings)
        assert app is not None

    def test_create_app_without_settings(self) -> None:
        app = create_app()
        assert app is not None

    def test_create_app_includes_evaluate_route(self) -> None:
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/v1/evaluate" in routes

    def test_create_app_includes_health_route(self) -> None:
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/health" in routes


# ---------------------------------------------------------------------------
# Lifespan — startup success and failure
# ---------------------------------------------------------------------------

class TestLifespan:
    def test_lifespan_startup_loads_pipeline_into_app_state(self) -> None:
        settings = _test_settings(enable_bert=False)

        with patch("clinical_safeguard.api.app.get_settings", return_value=settings):
            app = create_app()
            with TestClient(app) as client:
                assert hasattr(app.state, "safeguard_pipeline")
                response = client.get("/health")
                assert response.status_code == 200

    def test_lifespan_calls_hf_init_when_bert_enabled(self) -> None:
        settings = _test_settings(enable_bert=True, hf_token="hf_testtoken")

        with patch("clinical_safeguard.api.app.get_settings", return_value=settings), \
                patch("clinical_safeguard.api.app.initialize_hf_services") as mock_init:
            app = create_app()
            with TestClient(app):
                mock_init.assert_called_once_with(settings.hf_token)

    def test_lifespan_skips_hf_init_when_bert_disabled(self) -> None:
        settings = _test_settings(enable_bert=False)

        with patch("clinical_safeguard.api.app.get_settings", return_value=settings), \
                patch("clinical_safeguard.api.app.initialize_hf_services") as mock_init:
            app = create_app()
            with TestClient(app):
                mock_init.assert_not_called()

    def test_lifespan_startup_failure_raises_resource_load_error(
            self, tmp_path: Path
    ) -> None:
        bad_settings = make_settings(
            SAFEGUARD_CRISIS_KEYWORDS=str(tmp_path / "no.yaml"),
            SAFEGUARD_MALIGN_KEYWORDS=str(tmp_path / "no.yaml"),
            SAFEGUARD_BYPASS_PATTERNS=str(tmp_path / "no.yaml"),
            SAFEGUARD_ENABLE_BERT="false",
        )
        with patch("clinical_safeguard.api.app.get_settings", return_value=bad_settings):
            app = create_app()
            with pytest.raises(Exception):
                with TestClient(app, raise_server_exceptions=True):
                    pass


# ---------------------------------------------------------------------------
# main.py — just verify the module-level app is a FastAPI instance
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_app_is_importable(self) -> None:
        from main import app as main_app  # noqa: PLC0415
        from fastapi import FastAPI
        assert isinstance(main_app, FastAPI)
