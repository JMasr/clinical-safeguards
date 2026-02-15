from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ResourceLoadError, StageExecutionError
from src.models import Label, PromptInput
from src.stages.semantic import SemanticBERTStage, _DEFAULT_LABEL_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt(text: str) -> PromptInput:
    return PromptInput(text=text)


def _make_stage(
        model_output: list[dict] | None = None,
        factory_side_effect: Exception | None = None,
        threshold: float = 0.75,
        timeout: int = 5,
        label_map: dict | None = None,
) -> SemanticBERTStage:
    """
    Build a SemanticBERTStage with a mock pipeline_factory.
    No real model is ever loaded.
    """
    mock_classifier = MagicMock()
    if model_output is not None:
        mock_classifier.return_value = model_output

    if factory_side_effect is not None:
        mock_factory = MagicMock(side_effect=factory_side_effect)
    else:
        mock_factory = MagicMock(return_value=mock_classifier)

    return SemanticBERTStage(
        model_id="test/mock-model",
        threshold=threshold,
        inference_timeout_s=timeout,
        label_map=label_map,
        pipeline_factory=mock_factory,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_stage_name(self) -> None:
        stage = _make_stage(model_output=[{"label": "LABEL_0", "score": 0.9}])
        assert stage.name == "semantic_bert"

    def test_model_is_not_loaded_at_construction(self) -> None:
        """Lazy loading: factory must NOT be called during __init__."""
        mock_factory = MagicMock(return_value=MagicMock())
        SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=5,
            pipeline_factory=mock_factory,
        )
        mock_factory.assert_not_called()

    def test_default_label_map_used_when_none_provided(self) -> None:
        stage = SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=5,
            pipeline_factory=MagicMock(return_value=MagicMock()),
        )
        assert stage._label_map is _DEFAULT_LABEL_MAP


# ---------------------------------------------------------------------------
# Valid / safe outputs
# ---------------------------------------------------------------------------

class TestValidOutputs:
    def test_label0_above_threshold_returns_valid(self) -> None:
        stage = _make_stage(model_output=[{"label": "LABEL_0", "score": 0.95}])
        result = stage.process(_prompt("How can I improve my sleep?"))
        assert result.label == Label.VALID
        assert result.short_circuit is False

    def test_safe_label_returns_valid(self) -> None:
        stage = _make_stage(
            model_output=[{"label": "safe", "score": 0.99}],
            label_map={"SAFE": Label.VALID, "UNSAFE": Label.MALIGN},
        )
        result = stage.process(_prompt("Tell me about anxiety"))
        assert result.label == Label.VALID

    def test_confidence_score_preserved_in_result(self) -> None:
        stage = _make_stage(model_output=[{"label": "LABEL_0", "score": 0.87}])
        result = stage.process(_prompt("Hello"))
        assert result.confidence == pytest.approx(0.87)

    def test_triggered_by_contains_model_id_and_raw_label(self) -> None:
        stage = _make_stage(model_output=[{"label": "LABEL_0", "score": 0.9}])
        result = stage.process(_prompt("Hello"))
        assert "test/mock-model" in result.triggered_by
        assert "LABEL_0" in result.triggered_by

    def test_stage_name_in_result(self) -> None:
        stage = _make_stage(model_output=[{"label": "LABEL_0", "score": 0.9}])
        result = stage.process(_prompt("Hello"))
        assert result.stage_name == "semantic_bert"


# ---------------------------------------------------------------------------
# Crisis detection
# ---------------------------------------------------------------------------

class TestCrisisDetection:
    def test_label1_above_threshold_returns_crisis(self) -> None:
        stage = _make_stage(model_output=[{"label": "LABEL_1", "score": 0.92}])
        result = stage.process(_prompt("I want to end my life"))
        assert result.label == Label.CRISIS
        assert result.short_circuit is True

    def test_crisis_label_returns_crisis(self) -> None:
        stage = _make_stage(
            model_output=[{"label": "crisis", "score": 0.88}],
            label_map={"CRISIS": Label.CRISIS, "NORMAL": Label.VALID},
        )
        result = stage.process(_prompt("suicidal thoughts"))
        assert result.label == Label.CRISIS

    def test_suicide_label_returns_crisis(self) -> None:
        stage = _make_stage(
            model_output=[{"label": "suicide", "score": 0.95}],
        )
        result = stage.process(_prompt("test"))
        assert result.label == Label.CRISIS
        assert result.short_circuit is True


# ---------------------------------------------------------------------------
# Malign detection
# ---------------------------------------------------------------------------

class TestMalignDetection:
    def test_toxic_label_above_threshold_returns_malign(self) -> None:
        stage = _make_stage(model_output=[{"label": "toxic", "score": 0.91}])
        result = stage.process(_prompt("I want to hurt you"))
        assert result.label == Label.MALIGN
        assert result.short_circuit is True

    def test_unsafe_label_returns_malign(self) -> None:
        stage = _make_stage(
            model_output=[{"label": "unsafe", "score": 0.83}],
        )
        result = stage.process(_prompt("malicious input"))
        assert result.label == Label.MALIGN


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_crisis_below_threshold_downgrades_to_valid(self) -> None:
        stage = _make_stage(
            model_output=[{"label": "LABEL_1", "score": 0.60}],
            threshold=0.75,
        )
        result = stage.process(_prompt("I feel sad sometimes"))
        assert result.label == Label.VALID
        assert result.short_circuit is False

    def test_malign_below_threshold_downgrades_to_valid(self) -> None:
        stage = _make_stage(
            model_output=[{"label": "toxic", "score": 0.50}],
            threshold=0.75,
        )
        result = stage.process(_prompt("some ambiguous text"))
        assert result.label == Label.VALID

    def test_score_exactly_at_threshold_is_kept(self) -> None:
        """score >= threshold → label is kept (not downgraded)."""
        stage = _make_stage(
            model_output=[{"label": "LABEL_1", "score": 0.75}],
            threshold=0.75,
        )
        result = stage.process(_prompt("test"))
        assert result.label == Label.CRISIS

    def test_valid_label_below_threshold_remains_valid(self) -> None:
        """Threshold downgrade only applies to non-VALID labels."""
        stage = _make_stage(
            model_output=[{"label": "LABEL_0", "score": 0.30}],
            threshold=0.75,
        )
        result = stage.process(_prompt("Hello"))
        assert result.label == Label.VALID


# ---------------------------------------------------------------------------
# Unknown label handling
# ---------------------------------------------------------------------------

class TestUnknownLabels:
    def test_unknown_label_treated_as_valid(self) -> None:
        stage = _make_stage(model_output=[{"label": "UNKNOWN_LABEL", "score": 0.99}])
        result = stage.process(_prompt("test"))
        assert result.label == Label.VALID

    def test_label_lookup_is_case_insensitive(self) -> None:
        """label_map keys are upper-cased before lookup."""
        stage = _make_stage(
            model_output=[{"label": "label_1", "score": 0.90}],
        )
        result = stage.process(_prompt("test"))
        assert result.label == Label.CRISIS


# ---------------------------------------------------------------------------
# Lazy loading and thread safety
# ---------------------------------------------------------------------------

class TestLazyLoading:
    def test_model_loaded_on_first_process_call(self) -> None:
        mock_factory = MagicMock(return_value=MagicMock(
            return_value=[{"label": "LABEL_0", "score": 0.9}]
        ))
        stage = SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=5,
            pipeline_factory=mock_factory,
        )
        mock_factory.assert_not_called()
        stage.process(_prompt("Hello"))
        mock_factory.assert_called_once()

    def test_model_loaded_only_once_across_multiple_calls(self) -> None:
        mock_factory = MagicMock(return_value=MagicMock(
            return_value=[{"label": "LABEL_0", "score": 0.9}]
        ))
        stage = SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=5,
            pipeline_factory=mock_factory,
        )
        for _ in range(5):
            stage.process(_prompt("Hello"))
        assert mock_factory.call_count == 1

    def test_concurrent_calls_load_model_exactly_once(self) -> None:
        """Thread-safety: model must be loaded exactly once under concurrency."""
        load_count = 0
        load_lock = threading.Lock()

        def counting_factory(*args, **kwargs):
            nonlocal load_count
            time.sleep(0.01)  # simulate slow load
            with load_lock:
                load_count += 1
            return MagicMock(return_value=[{"label": "LABEL_0", "score": 0.9}])

        stage = SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=5,
            pipeline_factory=counting_factory,
        )

        threads = [
            threading.Thread(target=stage.process, args=(_prompt("Hello"),))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert load_count == 1


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_inference_timeout_raises_stage_execution_error(self) -> None:
        def slow_classifier(text):
            time.sleep(10)  # much longer than timeout
            return [{"label": "LABEL_0", "score": 0.9}]

        mock_factory = MagicMock(return_value=slow_classifier)
        stage = SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=1,
            pipeline_factory=mock_factory,
        )

        with pytest.raises(StageExecutionError) as exc_info:
            stage.process(_prompt("Hello"))

        assert exc_info.value.stage_name == "semantic_bert"
        assert isinstance(exc_info.value.cause, TimeoutError)

    def test_timeout_error_cause_mentions_timeout(self) -> None:
        def slow_classifier(text):
            time.sleep(10)

        mock_factory = MagicMock(return_value=slow_classifier)
        stage = SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=1,
            pipeline_factory=mock_factory,
        )

        with pytest.raises(StageExecutionError) as exc_info:
            stage.process(_prompt("Hello"))

        assert "timeout" in str(exc_info.value.cause).lower()


# ---------------------------------------------------------------------------
# Model load failures → fail-closed
# ---------------------------------------------------------------------------

class TestModelLoadFailures:
    def test_network_error_during_load_raises_resource_load_error(self) -> None:
        stage = _make_stage(
            factory_side_effect=OSError("Connection refused")
        )
        with pytest.raises(StageExecutionError):
            stage.process(_prompt("Hello"))

    def test_import_error_in_default_factory(self) -> None:
        """If transformers is not installed, default factory raises ResourceLoadError."""
        stage = SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=5,
        )
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(StageExecutionError):
                stage.process(_prompt("Hello"))

    def test_resource_load_error_propagates_as_stage_execution_error(self) -> None:
        """ResourceLoadError from _load_model is caught by process() fail-closed handler."""
        stage = _make_stage(factory_side_effect=MemoryError("CUDA OOM"))
        with pytest.raises(StageExecutionError) as exc_info:
            stage.process(_prompt("Hello"))
        assert exc_info.value.stage_name == "semantic_bert"

    def test_inference_exception_propagates_as_stage_execution_error(self) -> None:
        """If the classifier itself raises, it must become a StageExecutionError."""
        mock_classifier = MagicMock(side_effect=RuntimeError("CUDA error"))
        mock_factory = MagicMock(return_value=mock_classifier)
        stage = SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=5,
            pipeline_factory=mock_factory,
        )
        with pytest.raises(StageExecutionError) as exc_info:
            stage.process(_prompt("Hello"))
        assert exc_info.value.stage_name == "semantic_bert"


# ---------------------------------------------------------------------------
# Malformed model output
# ---------------------------------------------------------------------------

class TestMalformedOutput:
    def test_empty_list_output_raises_stage_execution_error(self) -> None:
        stage = _make_stage(model_output=[])
        with pytest.raises(StageExecutionError) as exc_info:
            stage.process(_prompt("Hello"))
        assert "Unexpected model output" in str(exc_info.value)

    def test_missing_label_key_raises_stage_execution_error(self) -> None:
        stage = _make_stage(model_output=[{"score": 0.9}])
        with pytest.raises(StageExecutionError):
            stage.process(_prompt("Hello"))

    def test_missing_score_key_raises_stage_execution_error(self) -> None:
        stage = _make_stage(model_output=[{"label": "LABEL_0"}])
        with pytest.raises(StageExecutionError):
            stage.process(_prompt("Hello"))

    def test_non_list_output_raises_stage_execution_error(self) -> None:
        stage = _make_stage(model_output={"label": "LABEL_0", "score": 0.9})  # type: ignore
        with pytest.raises(StageExecutionError):
            stage.process(_prompt("Hello"))


# ---------------------------------------------------------------------------
# StageExecutionError re-raise (not double-wrapped)
# ---------------------------------------------------------------------------

class TestStageExecutionErrorNotDoubleWrapped:
    def test_stage_execution_error_from_timeout_is_not_rewrapped(self) -> None:
        """A StageExecutionError from _run_with_timeout must propagate as-is."""

        def slow_classifier(text):
            time.sleep(10)

        mock_factory = MagicMock(return_value=slow_classifier)
        stage = SemanticBERTStage(
            model_id="test/model",
            threshold=0.75,
            inference_timeout_s=1,
            pipeline_factory=mock_factory,
        )

        with pytest.raises(StageExecutionError) as exc_info:
            stage.process(_prompt("Hello"))

        # Must be stage_name="semantic_bert" from _run_with_timeout,
        # not a new wrapping from process()'s outer except.
        assert exc_info.value.stage_name == "semantic_bert"
        assert isinstance(exc_info.value.cause, TimeoutError)


# ---------------------------------------------------------------------------
# Default pipeline factory
# ---------------------------------------------------------------------------

class TestDefaultPipelineFactory:
    def test_default_factory_calls_transformers_pipeline(self) -> None:
        """The default factory delegates to transformers.pipeline when available."""
        mock_pipeline_fn = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.9}])
        mock_transformers = MagicMock()
        mock_transformers.pipeline = mock_pipeline_fn

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            result = SemanticBERTStage._default_pipeline_factory(
                "text-classification", model="test/model", truncation=True
            )

        mock_pipeline_fn.assert_called_once_with(
            "text-classification", model="test/model", truncation=True
        )
        assert result is mock_pipeline_fn.return_value

    def test_default_factory_raises_resource_load_error_if_transformers_missing(self) -> None:
        """When transformers is not installed, ResourceLoadError is raised."""
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ResourceLoadError):
                SemanticBERTStage._default_pipeline_factory(
                    "text-classification", model="some/model"
                )
