from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import PromptInput, StageResult


class GuardrailStage(ABC):
    """
    Abstract base for every pipeline stage.

    Contract:
    - process() MUST return a StageResult. It must never raise — any
      internal exception should be caught and re-raised as StageExecutionError
      so the pipeline's fail-closed handler can intercept it uniformly.
    - Stages are stateless with respect to a single evaluation: they may hold
      pre-loaded resources (keyword lists, model handles) but must never
      mutate the PromptInput they receive.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in StageResult.stage_name and logs."""

    @abstractmethod
    def process(self, prompt: PromptInput) -> StageResult:
        """
        Evaluate the prompt and return a StageResult.

        Raises:
            StageExecutionError: wrapping any internal failure.
        """
