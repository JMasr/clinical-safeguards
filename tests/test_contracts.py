from __future__ import annotations

import pytest
from pydantic import ValidationError

from clinical_safeguard.config.settings import get_settings
from clinical_safeguard.core.exceptions import (
    ResourceLoadError,
    SafeguardError,
    StageExecutionError,
)
from clinical_safeguard.models import Label, ResponseCode
from clinical_safeguard.models.response import PromptInput, StageResult


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
        exc = StageExecutionError("s", ValueError("x"))
        assert isinstance(exc, SafeguardError)

    def test_resource_load_error_message(self) -> None:
        cause = FileNotFoundError("no file")
        exc = ResourceLoadError("keywords.yaml", cause)

        assert "keywords.yaml" in str(exc)
        assert "FileNotFoundError" in str(exc)
        assert exc.resource == "keywords.yaml"
        assert exc.cause is cause

    def test_resource_load_error_is_safeguard_error(self) -> None:
        exc = ResourceLoadError("res", IOError("x"))
        assert isinstance(exc, SafeguardError)


# ---------------------------------------------------------------------------
# PromptInput validation
# ---------------------------------------------------------------------------

class TestPromptInput:
    def test_strips_whitespace(self) -> None:
        p = PromptInput(text="  hello  ")
        assert p.text == "hello"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValidationError):
            PromptInput(text="")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValidationError):
            PromptInput(text="   ")

    def test_rejects_text_over_max_length(self) -> None:
        with pytest.raises(ValidationError):
            PromptInput(text="a" * 8193)

    def test_session_id_optional(self) -> None:
        p = PromptInput(text="hello")
        assert p.session_id is None

    def test_session_id_accepted(self) -> None:
        p = PromptInput(text="hello", session_id="abc-123")
        assert p.session_id == "abc-123"


# ---------------------------------------------------------------------------
# StageResult validation
# ---------------------------------------------------------------------------

class TestStageResult:
    def test_confidence_must_be_between_0_and_1(self) -> None:
        with pytest.raises(ValidationError):
            StageResult(stage_name="s", label=Label.VALID, confidence=1.1)

        with pytest.raises(ValidationError):
            StageResult(stage_name="s", label=Label.VALID, confidence=-0.1)

    def test_valid_stage_result(self) -> None:
        r = StageResult(stage_name="det", label=Label.CRISIS, confidence=0.95)
        assert r.label == Label.CRISIS
        assert r.short_circuit is False  # default


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

from tests.conftest import make_settings  # noqa: E402


class TestSettings:
    def test_defaults_are_valid_with_bert_disabled(self) -> None:
        """make_settings passes _env_file=None — no .env bleed-through."""
        s = make_settings(SAFEGUARD_ENABLE_BERT="false", HF_TOKEN=None)
        assert s.model_threshold == 0.75
        assert s.inference_timeout_s == 10
        assert s.enable_semantic_stage is False
        assert s.hf_token is None

    def test_bert_enabled_with_token_is_valid(self) -> None:
        s = make_settings(SAFEGUARD_ENABLE_BERT="true", HF_TOKEN="hf_testtoken123")
        assert s.enable_semantic_stage is True
        assert s.hf_token is not None
        assert "hf_testtoken123" not in repr(s.hf_token)

    def test_bert_enabled_without_token_raises_value_error(self) -> None:
        with pytest.raises(ValidationError, match="HF_TOKEN is required"):
            make_settings(SAFEGUARD_ENABLE_BERT="true", HF_TOKEN=None)

    def test_hf_token_secret_str_hides_value(self) -> None:
        s = make_settings(SAFEGUARD_ENABLE_BERT="false", HF_TOKEN="hf_supersecret")
        assert s.hf_token is not None
        assert "hf_supersecret" not in repr(s)
        assert s.hf_token.get_secret_value() == "hf_supersecret"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAFEGUARD_THRESHOLD", "0.9")
        s = make_settings(SAFEGUARD_ENABLE_BERT="false")
        assert s.model_threshold == 0.9
        assert s.enable_semantic_stage is False

    def test_get_settings_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAFEGUARD_ENABLE_BERT", "false")
        monkeypatch.setenv("HF_TOKEN", "")
        get_settings.cache_clear()
        try:
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2
        finally:
            get_settings.cache_clear()

    def test_get_settings_cache_clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAFEGUARD_ENABLE_BERT", "false")
        monkeypatch.setenv("HF_TOKEN", "")
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
        assert Label.VALID == "Válida"
        assert Label.MALIGN == "Maligna"
        assert Label.CRISIS == "Crisis"
        assert Label.ERROR == "Server Error"

    def test_response_code_values(self) -> None:
        assert ResponseCode.VALID == 100
        assert ResponseCode.MALIGN == 400
        assert ResponseCode.CRISIS == 406
        assert ResponseCode.ERROR == 500
