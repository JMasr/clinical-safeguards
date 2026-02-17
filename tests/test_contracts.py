from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import get_settings
from src.core import (
    ResourceLoadError,
    SafeguardError,
    StageExecutionError,
)
from src.models import Label, ResponseCode, PromptInput, StageResult
from tests.conftest import make_settings


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_stage_execution_error_message(self) -> None:
        cause = RuntimeError("OOM")
        exc = StageExecutionError("my_stage", cause)

        assert "my_stage" in str(exc)
        assert "RuntimeError" in str(exc)
        assert "OOM" in str(exc)
        assert exc.stage_name == "my_stage"
        assert exc.cause is cause

    def test_stage_execution_error_is_safeguard_error(self) -> None:
        exc = StageExecutionError("s", RuntimeError())
        assert isinstance(exc, SafeguardError)

    def test_resource_load_error_is_safeguard_error(self) -> None:
        exc = ResourceLoadError("missing file")
        assert isinstance(exc, SafeguardError)

    def test_resource_load_error_message(self) -> None:
        exc = ResourceLoadError("keywords.yaml not found")
        assert "keywords.yaml" in str(exc)


# ---------------------------------------------------------------------------
# PromptInput
# ---------------------------------------------------------------------------

class TestPromptInput:
    def test_valid_prompt(self) -> None:
        p = PromptInput(text="Hello world")
        assert p.text == "Hello world"

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            PromptInput(text="")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValidationError):
            PromptInput(text="   ")

    def test_text_is_stripped(self) -> None:
        p = PromptInput(text="  hello  ")
        assert p.text == "hello"

    def test_text_immutable(self) -> None:
        p = PromptInput(text="hello")
        with pytest.raises(Exception):
            p.text = "modified"


# ---------------------------------------------------------------------------
# StageResult
# ---------------------------------------------------------------------------

class TestStageResult:
    def test_valid_stage_result(self) -> None:
        r = StageResult(
            stage_name="test",
            label=Label.VALID,
            confidence=0.9,
        )
        assert r.stage_name == "test"
        assert r.label == Label.VALID
        assert r.short_circuit is False

    def test_confidence_must_be_between_0_and_1(self) -> None:
        with pytest.raises(ValidationError):
            StageResult(stage_name="s", label=Label.VALID, confidence=1.5)


# ---------------------------------------------------------------------------
# Settings — post-Hydra refactor
# Pipeline composition fields (enable_*, model_id, attack_model_*)
# have moved to Hydra configs. Settings only holds secrets + infra.
# ---------------------------------------------------------------------------

class TestSettings:
    def test_defaults_no_token(self) -> None:
        s = make_settings(HF_TOKEN=None)
        assert s.hf_token is None
        assert s.inference_timeout_s == 10

    def test_hf_token_loaded(self) -> None:
        s = make_settings(HF_TOKEN="hf_testtoken123")
        assert s.hf_token is not None

    def test_hf_token_secret_str_hides_value(self) -> None:
        s = make_settings(HF_TOKEN="hf_supersecret")
        assert "hf_supersecret" not in repr(s)
        assert s.hf_token.get_secret_value() == "hf_supersecret"

    def test_timeout_override(self) -> None:
        s = make_settings(SAFEGUARD_TIMEOUT="30")
        assert s.inference_timeout_s == 30

    def test_env_override_via_monkeypatch(
            self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SAFEGUARD_TIMEOUT", "42")
        s = make_settings()
        assert s.inference_timeout_s == 42

    def test_settings_no_longer_has_enable_bert_field(self) -> None:
        """Pipeline flags moved to Hydra — must not exist on Settings."""
        s = make_settings()
        assert not hasattr(s, "enable_semantic_stage")
        assert not hasattr(s, "enable_attack_stage")

    def test_settings_no_longer_has_model_id_field(self) -> None:
        """Model IDs moved to Hydra YAML — must not exist on Settings."""
        s = make_settings()
        assert not hasattr(s, "model_id")
        assert not hasattr(s, "attack_model_id")

    def test_get_settings_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAFEGUARD_TIMEOUT", "5")
        get_settings.cache_clear()
        try:
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2
        finally:
            get_settings.cache_clear()

    def test_get_settings_cache_clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAFEGUARD_TIMEOUT", "5")
        get_settings.cache_clear()
        try:
            s1 = get_settings()
            get_settings.cache_clear()
            s2 = get_settings()
            assert s1 is not s2
        finally:
            get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_label_values(self) -> None:
        assert Label.VALID == "Valid"
        assert Label.MALIGN == "Malign"
        assert Label.CRISIS == "Crisis"
        assert Label.ERROR == "Server Error"

    def test_response_code_values(self) -> None:
        assert ResponseCode.VALID == 100
        assert ResponseCode.MALIGN == 400
        assert ResponseCode.CRISIS == 406
        assert ResponseCode.ERROR == 500