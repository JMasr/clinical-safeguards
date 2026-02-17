from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

from src.core.base import GuardrailStage
from src.core.exceptions import StageExecutionError
from src.models import (
    FinalResponse,
    Label,
    LABEL_PRECEDENCE,
    LABEL_TO_CODE,
    PromptInput,
    ResponseData,
    StageResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageTrace:
    """
    Internal execution record for a single stage run.

    Intentionally NOT a Pydantic model — this is an implementation detail
    of SafeguardPipeline, not a public contract. The HTTP layer translates
    StageTrace → StageTraceResponse (Pydantic) before serialization.

    This separation lets the pipeline evolve freely (e.g. parallel execution,
    retries, conditional branching) without breaking external consumers.
    A parallel execution would produce StageTrace objects with overlapping
    wall-clock times; the HTTP adapter decides how to present that.
    """

    result: StageResult
    duration_ms: float


@dataclass(frozen=True)
class PipelineTrace:
    """
    Full execution record returned by evaluate_with_trace().
    Groups all StageTrace objects with the aggregate timing.
    """

    stage_traces: tuple[StageTrace, ...]
    total_duration_ms: float
    final_response: FinalResponse


class SafeguardPipeline:
    """
    Orchestrates a sequence of GuardrailStage instances.

    Execution rules:
    1. Stages run in the order they were registered.
    2. If a stage returns short_circuit=True (always the case for Crisis /
       Maligna), the pipeline stops immediately — no subsequent stage runs.
    3. If all stages complete without short-circuiting, the result with the
       highest label precedence wins (Crisis > Maligna > Server Error > Válida).
    4. FAIL-CLOSED: any unhandled exception inside evaluate() — including
       StageExecutionError — produces a Server Error response. The original
       prompt text is never included in error responses.

    Two public entry points:
      - evaluate()            → FinalResponse only (production path)
      - evaluate_with_trace() → PipelineTrace (inspect/debug path)
    """

    def __init__(self, stages: Sequence[GuardrailStage]) -> None:
        if not stages:
            raise ValueError("Pipeline must have at least one stage.")
        self._stages: tuple[GuardrailStage, ...] = tuple(stages)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, prompt: PromptInput) -> FinalResponse:
        """
        Production entry point. Always returns a FinalResponse — never raises.
        Timing overhead is not recorded — use evaluate_with_trace() for that.
        """
        try:
            traces = self._run_stages(prompt)
            winning = self._merge_results([t.result for t in traces])
            return self._build_response(prompt, winning)
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            logger.exception(
                "Unhandled exception in pipeline. Returning fail-closed response.",
                exc_info=exc,
            )
            return self._fail_closed_response()

    def evaluate_with_trace(self, prompt: PromptInput) -> PipelineTrace:
        """
        Inspect/debug entry point. Returns the full execution trace including
        per-stage timing and the final response.

        Shares the same fail-closed guarantee as evaluate(): exceptions are
        caught and represented as a Server Error PipelineTrace, never raised.

        Only exposed via /v1/inspect (requires SAFEGUARD_INSPECT_MODE=true).
        Never call this from the production /v1/evaluate path — timing
        instrumentation adds overhead and the trace data is not needed.
        """
        t_start = time.perf_counter()
        try:
            traces = self._run_stages(prompt)
            winning = self._merge_results([t.result for t in traces])
            final = self._build_response(prompt, winning)
            total_ms = (time.perf_counter() - t_start) * 1000
            return PipelineTrace(
                stage_traces=tuple(traces),
                total_duration_ms=round(total_ms, 3),
                final_response=final,
            )
        except Exception as exc:  # noqa: BLE001
            total_ms = (time.perf_counter() - t_start) * 1000
            logger.exception(
                "Unhandled exception in pipeline (trace mode). Returning fail-closed.",
                exc_info=exc,
            )
            return PipelineTrace(
                stage_traces=(),
                total_duration_ms=round(total_ms, 3),
                final_response=self._fail_closed_response(),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_stages(self, prompt: PromptInput) -> list[StageTrace]:
        """
        Execute stages sequentially. Stops on short_circuit=True.
        Returns a StageTrace per executed stage (not-executed stages are absent).
        Propagates StageExecutionError so the outer try/except can catch it.
        """
        traces: list[StageTrace] = []

        for stage in self._stages:
            logger.debug("Running stage: %s", stage.name)

            t0 = time.perf_counter()
            try:
                result = stage.process(prompt)
            except StageExecutionError:
                raise
            except Exception as exc:
                raise StageExecutionError(stage.name, exc) from exc
            finally:
                # duration is always recorded, even on error paths
                pass

            duration_ms = round((time.perf_counter() - t0) * 1000, 3)
            trace = StageTrace(result=result, duration_ms=duration_ms)
            traces.append(trace)

            logger.debug(
                "Stage %s → label=%s confidence=%.3f short_circuit=%s duration_ms=%.1f",
                stage.name,
                result.label,
                result.confidence,
                result.short_circuit,
                duration_ms,
            )

            if result.short_circuit:
                logger.info(
                    "Short-circuit triggered by stage '%s' with label '%s'.",
                    stage.name,
                    result.label,
                )
                break

        return traces

    @staticmethod
    def _merge_results(results: list[StageResult]) -> StageResult:
        """
        Return the result with the highest label precedence.
        Guaranteed non-empty because the pipeline has ≥1 stage.
        """
        return max(results, key=lambda r: LABEL_PRECEDENCE[r.label])

    @staticmethod
    def _build_response(prompt: PromptInput, result: StageResult) -> FinalResponse:
        metadatos: dict = {"stage": result.stage_name}
        if result.triggered_by:
            metadatos["triggered_by"] = result.triggered_by

        return FinalResponse(
            code=LABEL_TO_CODE[result.label],
            etiqueta=result.label,
            data=ResponseData(
                texto_procesado=prompt.text,
                score_confianza=result.confidence,
                metadatos=metadatos,
            ),
        )

    @staticmethod
    def _fail_closed_response() -> FinalResponse:
        """
        Returned on any unhandled exception.
        Does NOT include the original prompt text — intentional.
        """
        return FinalResponse(
            code=LABEL_TO_CODE[Label.ERROR],
            etiqueta=Label.ERROR,
            data=ResponseData(
                texto_procesado="",
                score_confianza=0.0,
                metadatos={"reason": "Error de integridad del sistema"},
            ),
        )