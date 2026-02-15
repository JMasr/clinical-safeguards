from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from clinical_safeguard.core import (
    GuardrailStage,
    SafeguardPipeline,
    StageExecutionError,
)
from clinical_safeguard.models import (
    FinalResponse,
    Label,
    PromptInput,
    ResponseCode,
    StageResult,
)
from tests.conftest import make_stage_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stage(name: str, result: StageResult) -> GuardrailStage:
    """Build a mock stage that returns a fixed StageResult."""
    stage = MagicMock(spec=GuardrailStage)
    stage.name = name
    stage.process.return_value = result
    return stage


def _make_failing_stage(name: str, exc: Exception) -> GuardrailStage:
    """Build a mock stage that raises on process()."""
    stage = MagicMock(spec=GuardrailStage)
    stage.name = name
    stage.process.side_effect = exc
    return stage


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    def test_raises_if_no_stages(self) -> None:
        with pytest.raises(ValueError, match="at least one stage"):
            SafeguardPipeline(stages=[])

    def test_accepts_single_stage(self, valid_prompt: PromptInput) -> None:
        stage = _make_stage("s1", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[stage])
        response = pipeline.evaluate(valid_prompt)
        assert response.etiqueta == Label.VALID


# ---------------------------------------------------------------------------
# Happy path — Válida
# ---------------------------------------------------------------------------

class TestPipelineValidPath:
    def test_single_stage_valid(self, valid_prompt: PromptInput) -> None:
        stage = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[stage])

        response = pipeline.evaluate(valid_prompt)

        assert isinstance(response, FinalResponse)
        assert response.code == ResponseCode.VALID
        assert response.etiqueta == Label.VALID
        assert response.data.texto_procesado == valid_prompt.text
        assert response.data.score_confianza == 0.99

    def test_two_stages_both_valid(self, valid_prompt: PromptInput) -> None:
        s1 = _make_stage("det", make_stage_result(Label.VALID, confidence=0.8, short_circuit=False))
        s2 = _make_stage("bert", make_stage_result(Label.VALID, confidence=0.9, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        response = pipeline.evaluate(valid_prompt)

        assert response.etiqueta == Label.VALID
        # Both stages ran
        s1.process.assert_called_once()
        s2.process.assert_called_once()


# ---------------------------------------------------------------------------
# Short-circuit behaviour
# ---------------------------------------------------------------------------

class TestShortCircuit:
    def test_crisis_stops_pipeline(self, crisis_prompt: PromptInput) -> None:
        s1 = _make_stage(
            "det",
            make_stage_result(Label.CRISIS, short_circuit=True),
        )
        s2 = _make_stage(
            "bert",
            make_stage_result(Label.VALID, short_circuit=False),
        )
        pipeline = SafeguardPipeline(stages=[s1, s2])

        response = pipeline.evaluate(crisis_prompt)

        assert response.etiqueta == Label.CRISIS
        assert response.code == ResponseCode.CRISIS
        s1.process.assert_called_once()
        s2.process.assert_not_called()  # ← critical assertion

    def test_malign_stops_pipeline(self, malign_prompt: PromptInput) -> None:
        s1 = _make_stage(
            "det",
            make_stage_result(Label.MALIGN, short_circuit=True),
        )
        s2 = _make_stage("bert", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        response = pipeline.evaluate(malign_prompt)

        assert response.etiqueta == Label.MALIGN
        assert response.code == ResponseCode.MALIGN
        s2.process.assert_not_called()

    def test_valid_does_not_short_circuit(self, valid_prompt: PromptInput) -> None:
        s1 = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        s2 = _make_stage("bert", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        pipeline.evaluate(valid_prompt)

        s1.process.assert_called_once()
        s2.process.assert_called_once()


# ---------------------------------------------------------------------------
# Label precedence (no short-circuit, both stages complete)
# ---------------------------------------------------------------------------

class TestLabelPrecedence:
    def test_crisis_beats_malign(self, valid_prompt: PromptInput) -> None:
        s1 = _make_stage("det", make_stage_result(Label.MALIGN, short_circuit=False))
        s2 = _make_stage("bert", make_stage_result(Label.CRISIS, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        response = pipeline.evaluate(valid_prompt)

        assert response.etiqueta == Label.CRISIS

    def test_malign_beats_valid(self, valid_prompt: PromptInput) -> None:
        s1 = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        s2 = _make_stage("bert", make_stage_result(Label.MALIGN, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        response = pipeline.evaluate(valid_prompt)

        assert response.etiqueta == Label.MALIGN

    def test_error_beats_valid_but_loses_to_malign(self, valid_prompt: PromptInput) -> None:
        s1 = _make_stage("det", make_stage_result(Label.ERROR, short_circuit=False))
        s2 = _make_stage("bert", make_stage_result(Label.MALIGN, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        response = pipeline.evaluate(valid_prompt)

        assert response.etiqueta == Label.MALIGN


# ---------------------------------------------------------------------------
# Fail-closed — stage raises StageExecutionError
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_stage_execution_error_returns_server_error(
            self, valid_prompt: PromptInput
    ) -> None:
        exc = StageExecutionError("det", RuntimeError("model OOM"))
        s1 = _make_failing_stage("det", exc)
        pipeline = SafeguardPipeline(stages=[s1])

        response = pipeline.evaluate(valid_prompt)

        assert response.etiqueta == Label.ERROR
        assert response.code == ResponseCode.ERROR
        assert response.data.texto_procesado == ""  # never leaks prompt
        assert "Error de integridad" in response.data.metadatos["reason"]

    def test_unexpected_exception_returns_server_error(
            self, valid_prompt: PromptInput
    ) -> None:
        s1 = _make_failing_stage("det", RuntimeError("unexpected crash"))
        pipeline = SafeguardPipeline(stages=[s1])

        response = pipeline.evaluate(valid_prompt)

        assert response.etiqueta == Label.ERROR
        assert response.data.texto_procesado == ""

    def test_fail_closed_does_not_raise(self, valid_prompt: PromptInput) -> None:
        """evaluate() must never propagate exceptions to the caller."""
        s1 = _make_failing_stage("det", Exception("total failure"))
        pipeline = SafeguardPipeline(stages=[s1])

        # Should not raise under any circumstance
        response = pipeline.evaluate(valid_prompt)
        assert response is not None

    def test_second_stage_failure_still_fail_closed(
            self, valid_prompt: PromptInput
    ) -> None:
        s1 = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        s2 = _make_failing_stage("bert", TimeoutError("inference timeout"))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        response = pipeline.evaluate(valid_prompt)

        assert response.etiqueta == Label.ERROR
        assert response.data.texto_procesado == ""


# ---------------------------------------------------------------------------
# Response structure integrity
# ---------------------------------------------------------------------------

class TestResponseStructure:
    def test_metadatos_contains_stage_name(self, valid_prompt: PromptInput) -> None:
        result = make_stage_result(Label.VALID, stage_name="deterministic", short_circuit=False)
        stage = _make_stage("deterministic", result)
        pipeline = SafeguardPipeline(stages=[stage])

        response = pipeline.evaluate(valid_prompt)

        assert response.data.metadatos["stage"] == "deterministic"

    def test_metadatos_contains_triggered_by_when_present(
            self, crisis_prompt: PromptInput
    ) -> None:
        result = make_stage_result(
            Label.CRISIS,
            stage_name="deterministic",
            triggered_by="keyword:suicidio",
            short_circuit=True,
        )
        stage = _make_stage("deterministic", result)
        pipeline = SafeguardPipeline(stages=[stage])

        response = pipeline.evaluate(crisis_prompt)

        assert response.data.metadatos["triggered_by"] == "keyword:suicidio"

    def test_texto_procesado_equals_input_text(self, valid_prompt: PromptInput) -> None:
        stage = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[stage])

        response = pipeline.evaluate(valid_prompt)

        assert response.data.texto_procesado == valid_prompt.text

    def test_input_immutability(self, valid_prompt: PromptInput) -> None:
        """PromptInput must be frozen — stages cannot mutate it."""
        with pytest.raises(Exception):
            valid_prompt.text = "mutated"  # type: ignore[misc]
