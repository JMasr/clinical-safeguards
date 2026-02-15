from __future__ import annotations

import logging
import threading
from typing import Any

from src.core.base import GuardrailStage
from src.core.exceptions import ResourceLoadError, StageExecutionError
from src.models import Label, PromptInput, StageResult

logger = logging.getLogger(__name__)

# HuggingFace label strings that map to each of our domain labels.
# These are the default output labels from models fine-tuned on mental-health
# or toxicity classification tasks. Override via LABEL_MAP if your model
# uses different strings.
#
# References for default label conventions:
#   - mental/mental-bert-base-uncased uses: "LABEL_0" (normal) / "LABEL_1" (mental health concern)
#   - prosusai/finbert, j-hartmann/emotion-english-distilroberta-base, etc.
#     use task-specific strings.
# The map below normalises whatever the model returns into our Label enum.
_DEFAULT_LABEL_MAP: dict[str, Label] = {
    # mental-bert style
    "LABEL_0": Label.VALID,
    "LABEL_1": Label.CRISIS,
    # common toxicity / safety classifiers
    "SAFE": Label.VALID,
    "UNSAFE": Label.MALIGN,
    "TOXIC": Label.MALIGN,
    "NORMAL": Label.VALID,
    # explicit crisis/suicide classifiers (e.g. gooofy/suicide-risk-bert)
    "SUICIDE": Label.CRISIS,
    "CRISIS": Label.CRISIS,
}


class SemanticBERTStage(GuardrailStage):
    """
    Phase 2: semantic classification via a HuggingFace text-classification model.

    Design decisions:
    - LAZY LOADING: the model is not loaded at construction time. It is loaded
      on the first call to process() and cached. This allows the service to
      start and pass health checks before the (potentially large) model is
      pulled into memory.
    - THREAD-SAFE INITIALIZATION: a threading.Lock ensures the model is
      loaded exactly once even under concurrent requests at startup.
    - TIMEOUT: inference runs in a background thread. If it exceeds
      `inference_timeout_s` the stage raises StageExecutionError, which the
      pipeline's fail-closed handler converts to a Server Error response.
    - MODEL SWAP: the model_id, threshold, label_map, and timeout are all
      injected at construction. No business logic changes are required to
      swap models.
    - TRUNCATION: inputs longer than the model's max token length are
      automatically truncated by the HuggingFace pipeline (truncation=True).

    Fail-closed paths (all → StageExecutionError → pipeline returns 500):
      - Model load failure (network, disk, OOM)
      - Inference timeout
      - Unexpected output format from the model
      - Any unhandled runtime exception
    """

    _NAME = "semantic_bert"

    def __init__(
            self,
            model_id: str,
            threshold: float,
            inference_timeout_s: int,
            label_map: dict[str, Label] | None = None,
            # Injectable factory — replaced by mocks in tests without touching
            # any business logic. Default is the real transformers.pipeline.
            pipeline_factory: Any = None,
    ) -> None:
        self._model_id = model_id
        self._threshold = threshold
        self._inference_timeout_s = inference_timeout_s
        self._label_map: dict[str, Label] = label_map or _DEFAULT_LABEL_MAP
        self._pipeline_factory = pipeline_factory or self._default_pipeline_factory

        # Lazy-loaded classifier — None until first use
        self._classifier: Any | None = None
        self._load_lock = threading.Lock()

        logger.info(
            "SemanticBERTStage configured — model=%s threshold=%.2f timeout=%ds",
            self._model_id,
            self._threshold,
            self._inference_timeout_s,
        )

    # ------------------------------------------------------------------
    # GuardrailStage interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._NAME

    def process(self, prompt: PromptInput) -> StageResult:
        try:
            classifier = self._get_classifier()
            raw_output = self._run_with_timeout(classifier, prompt.text)
            return self._parse_output(raw_output)
        except StageExecutionError:
            raise
        except Exception as exc:
            raise StageExecutionError(self._NAME, exc) from exc

    # ------------------------------------------------------------------
    # Model loading (lazy + thread-safe)
    # ------------------------------------------------------------------

    def _get_classifier(self) -> Any:
        """
        Return the cached classifier, loading it on first call.
        Uses double-checked locking for thread safety without paying the
        lock cost on every hot-path invocation.
        """
        if self._classifier is not None:
            return self._classifier

        with self._load_lock:
            # Second check inside the lock — another thread may have
            # loaded the model while we were waiting.
            if self._classifier is None:
                self._classifier = self._load_model()

        return self._classifier

    def _load_model(self) -> Any:
        """Load and return the HuggingFace pipeline. Wraps all errors."""
        logger.info("Loading model '%s'…", self._model_id)
        try:
            classifier = self._pipeline_factory(
                "text-classification",
                model=self._model_id,
                truncation=True,
            )
            logger.info("Model '%s' loaded successfully.", self._model_id)
            return classifier
        except Exception as exc:
            raise ResourceLoadError(self._model_id, exc) from exc

    # ------------------------------------------------------------------
    # Inference with timeout
    # ------------------------------------------------------------------

    def _run_with_timeout(self, classifier: Any, text: str) -> list[dict]:
        """
        Run inference in a background thread with a hard timeout.
        If the thread does not complete within `inference_timeout_s`, raises
        StageExecutionError so the pipeline's fail-closed handler activates.

        Note: Python threads cannot be forcibly killed. The background thread
        may still be running after timeout, but it is daemonized so it will
        not prevent process shutdown. This is an acceptable trade-off for
        a stateless inference workload.
        """
        result_container: list[Any] = []
        exc_container: list[Exception] = []

        def _inference_task() -> None:
            try:
                output = classifier(text)
                result_container.append(output)
            except Exception as exc:  # noqa: BLE001
                exc_container.append(exc)

        thread = threading.Thread(target=_inference_task, daemon=True)
        thread.start()
        thread.join(timeout=self._inference_timeout_s)

        if thread.is_alive():
            raise StageExecutionError(
                self._NAME,
                TimeoutError(
                    f"Inference on model '{self._model_id}' exceeded "
                    f"{self._inference_timeout_s}s timeout."
                ),
            )

        if exc_container:
            raise StageExecutionError(self._NAME, exc_container[0]) from exc_container[0]

        return result_container[0]

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, raw_output: list[dict]) -> StageResult:
        """
        Normalise the HuggingFace pipeline output to a StageResult.

        HuggingFace text-classification returns:
            [{"label": "LABEL_1", "score": 0.97}]  (single input)

        This method:
        1. Validates the output shape.
        2. Maps the model label string to our Label enum via label_map.
        3. Applies the confidence threshold: if score < threshold and the
           mapped label is non-VALID, we downgrade to VALID (the model is
           not confident enough to flag this input).
        4. Sets short_circuit=True for Crisis and Maligna — consistent with
           the deterministic stage contract.
        """
        try:
            prediction = raw_output[0]
            model_label: str = prediction["label"]
            score: float = float(prediction["score"])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise StageExecutionError(
                self._NAME,
                ValueError(f"Unexpected model output format: {raw_output!r}"),
            ) from exc

        # Map raw model label → domain Label
        mapped_label = self._label_map.get(model_label.upper(), None)
        if mapped_label is None:
            # Unknown label from model — log a warning and treat conservatively
            logger.warning(
                "Model '%s' returned unknown label '%s'. Treating as VALID. "
                "Update label_map if this label is expected.",
                self._model_id,
                model_label,
            )
            mapped_label = Label.VALID

        # Apply threshold: below threshold → downgrade to VALID
        if score < self._threshold and mapped_label != Label.VALID:
            logger.debug(
                "Score %.3f below threshold %.3f for label '%s' — downgrading to VALID.",
                score,
                self._threshold,
                mapped_label,
            )
            mapped_label = Label.VALID

        is_blocking = mapped_label in (Label.CRISIS, Label.MALIGN)

        return StageResult(
            stage_name=self._NAME,
            label=mapped_label,
            confidence=score,
            triggered_by=f"model:{self._model_id}|raw_label:{model_label}",
            short_circuit=is_blocking,
        )

    # ------------------------------------------------------------------
    # Default pipeline factory (the real HuggingFace import)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_pipeline_factory(task: str, **kwargs: Any) -> Any:
        """
        Deferred import of transformers.pipeline.
        Importing at module level would require transformers to be installed
        even when only running unit tests with mocks.
        """
        try:
            from transformers import pipeline  # noqa: PLC0415
            return pipeline(task, **kwargs)
        except ImportError as exc:
            raise ResourceLoadError(
                "transformers",
                ImportError("transformers library is not installed. Run: pip install transformers torch"),
            ) from exc
