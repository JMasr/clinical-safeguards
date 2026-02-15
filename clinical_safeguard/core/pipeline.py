from __future__ import annotations

import logging
from typing import Sequence

from clinical_safeguard.core.base import GuardrailStage
from clinical_safeguard.core.exceptions import StageExecutionError
from clinical_safeguard.models import (
    FinalResponse,
    Label,
    LABEL_PRECEDENCE,
    LABEL_TO_CODE,
    PromptInput,
    ResponseData,
    StageResult,
)

logger = logging.getLogger(__name__)


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
        Entry point. Always returns a FinalResponse — never raises.
        """
        try:
            results = self._run_stages(prompt)
            winning = self._merge_results(results)
            return self._build_response(prompt, winning)
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            logger.exception(
                "Unhandled exception in pipeline. Returning fail-closed response.",
                exc_info=exc,
            )
            return self._fail_closed_response()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_stages(self, prompt: PromptInput) -> list[StageResult]:
        """
        Execute stages sequentially. Stops on short_circuit=True.
        Propagates StageExecutionError so the outer try/except can catch it.
        """
        results: list[StageResult] = []

        for stage in self._stages:
            logger.debug("Running stage: %s", stage.name)
            try:
                result = stage.process(prompt)
            except StageExecutionError:
                # Let fail-closed handler at the top level deal with it.
                raise
            except Exception as exc:
                # A stage forgot to wrap its exception — we do it here.
                raise StageExecutionError(stage.name, exc) from exc

            results.append(result)
            logger.debug(
                "Stage %s → label=%s confidence=%.3f short_circuit=%s",
                stage.name,
                result.label,
                result.confidence,
                result.short_circuit,
            )

            if result.short_circuit:
                logger.info(
                    "Short-circuit triggered by stage '%s' with label '%s'.",
                    stage.name,
                    result.label,
                )
                break

        return results

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
