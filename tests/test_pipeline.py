from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core import (
    GuardrailStage,
    SafeguardPipeline,
    StageExecutionError,
)
from src.models import (
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


# ---------------------------------------------------------------------------
# stage_names property
# ---------------------------------------------------------------------------

class TestStageNames:
    def test_stage_names_returns_all_names_in_order(self, valid_prompt: PromptInput) -> None:
        s1 = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        s2 = _make_stage("bert", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        assert pipeline.stage_names == ("det", "bert")

    def test_stage_names_single_stage(self, valid_prompt: PromptInput) -> None:
        stage = _make_stage("only", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[stage])

        assert pipeline.stage_names == ("only",)

    def test_stage_names_includes_stages_not_run(self, valid_prompt: PromptInput) -> None:
        """stage_names reflects registered stages, not executed stages."""
        s1 = _make_stage("det", make_stage_result(Label.CRISIS, short_circuit=True))
        s2 = _make_stage("bert", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        pipeline.evaluate(valid_prompt)  # bert never runs
        assert pipeline.stage_names == ("det", "bert")  # still listed


# ---------------------------------------------------------------------------
# StageTrace / PipelineTrace — evaluate_with_trace()
# ---------------------------------------------------------------------------

from src.core.pipeline import PipelineTrace  # noqa: E402


class TestEvaluateWithTrace:
    def test_returns_pipeline_trace(self, valid_prompt: PromptInput) -> None:
        stage = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[stage])

        result = pipeline.evaluate_with_trace(valid_prompt)

        assert isinstance(result, PipelineTrace)

    def test_final_response_matches_evaluate(self, valid_prompt: PromptInput) -> None:
        stage = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[stage])

        trace = pipeline.evaluate_with_trace(valid_prompt)
        direct = pipeline.evaluate(valid_prompt)

        assert trace.final_response.etiqueta == direct.etiqueta
        assert trace.final_response.code == direct.code

    def test_trace_contains_one_entry_per_executed_stage(
            self, valid_prompt: PromptInput
    ) -> None:
        s1 = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        s2 = _make_stage("bert", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        result = pipeline.evaluate_with_trace(valid_prompt)

        assert len(result.stage_traces) == 2

    def test_short_circuit_limits_trace_entries(self, crisis_prompt: PromptInput) -> None:
        """Skipped stages must be absent from the trace — not present with empty data."""
        s1 = _make_stage("det", make_stage_result(Label.CRISIS, stage_name="det", short_circuit=True))
        s2 = _make_stage("bert", make_stage_result(Label.VALID, stage_name="bert", short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        result = pipeline.evaluate_with_trace(crisis_prompt)

        assert len(result.stage_traces) == 1
        assert result.stage_traces[0].result.stage_name == "det"

    def test_stage_trace_has_positive_duration(self, valid_prompt: PromptInput) -> None:
        stage = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[stage])

        result = pipeline.evaluate_with_trace(valid_prompt)

        assert result.stage_traces[0].duration_ms >= 0.0

    def test_total_duration_gte_sum_of_stages(self, valid_prompt: PromptInput) -> None:
        s1 = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        s2 = _make_stage("bert", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2])

        result = pipeline.evaluate_with_trace(valid_prompt)

        stage_sum = sum(t.duration_ms for t in result.stage_traces)
        assert result.total_duration_ms >= stage_sum

    def test_trace_preserves_stage_order(self, valid_prompt: PromptInput) -> None:
        s1 = _make_stage("first", make_stage_result(Label.VALID, stage_name="first", short_circuit=False))
        s2 = _make_stage("second", make_stage_result(Label.VALID, stage_name="second", short_circuit=False))
        s3 = _make_stage("third", make_stage_result(Label.VALID, stage_name="third", short_circuit=False))
        pipeline = SafeguardPipeline(stages=[s1, s2, s3])

        result = pipeline.evaluate_with_trace(valid_prompt)

        names = [t.result.stage_name for t in result.stage_traces]
        assert names == ["first", "second", "third"]

    def test_trace_stage_fields_match_stage_result(self, valid_prompt: PromptInput) -> None:
        sr = make_stage_result(
            Label.VALID,
            stage_name="deterministic",
            confidence=0.88,
            triggered_by="keyword:test",
            short_circuit=False,
        )
        stage = _make_stage("deterministic", sr)
        pipeline = SafeguardPipeline(stages=[stage])

        result = pipeline.evaluate_with_trace(valid_prompt)
        st = result.stage_traces[0]

        assert st.result.label == Label.VALID
        assert st.result.confidence == pytest.approx(0.88)
        assert st.result.triggered_by == "keyword:test"
        assert st.result.short_circuit is False

    def test_fail_closed_on_stage_error_returns_pipeline_trace(
            self, valid_prompt: PromptInput
    ) -> None:
        """evaluate_with_trace must never raise — same guarantee as evaluate()."""
        s1 = _make_failing_stage("det", RuntimeError("OOM"))
        pipeline = SafeguardPipeline(stages=[s1])

        result = pipeline.evaluate_with_trace(valid_prompt)

        assert isinstance(result, PipelineTrace)
        assert result.final_response.etiqueta == Label.ERROR
        assert result.final_response.data.texto_procesado == ""

    def test_fail_closed_trace_is_empty_on_error(self, valid_prompt: PromptInput) -> None:
        """On pipeline failure the trace tuple is empty — no partial data leaked."""
        s1 = _make_failing_stage("det", RuntimeError("crash"))
        pipeline = SafeguardPipeline(stages=[s1])

        result = pipeline.evaluate_with_trace(valid_prompt)

        assert result.stage_traces == ()

    def test_fail_closed_total_duration_recorded_even_on_error(
            self, valid_prompt: PromptInput
    ) -> None:
        s1 = _make_failing_stage("det", RuntimeError("crash"))
        pipeline = SafeguardPipeline(stages=[s1])

        result = pipeline.evaluate_with_trace(valid_prompt)

        assert result.total_duration_ms >= 0.0

    def test_stage_trace_is_frozen(self, valid_prompt: PromptInput) -> None:
        stage = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[stage])

        result = pipeline.evaluate_with_trace(valid_prompt)
        st = result.stage_traces[0]

        with pytest.raises(Exception):
            st.duration_ms = 999.0  # type: ignore[misc]

    def test_pipeline_trace_is_frozen(self, valid_prompt: PromptInput) -> None:
        stage = _make_stage("det", make_stage_result(Label.VALID, short_circuit=False))
        pipeline = SafeguardPipeline(stages=[stage])

        result = pipeline.evaluate_with_trace(valid_prompt)

        with pytest.raises(Exception):
            result.total_duration_ms = 999.0  # type: ignore[misc]