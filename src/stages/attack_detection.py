from __future__ import annotations

"""
AttackDetectionStage — semantic classifier for LLM attack detection.

Model: ProtectAI/deberta-v3-base-prompt-injection-v2
  - Architecture: DeBERTa-v3-base fine-tuned on prompt injection datasets
  - License:      Apache 2.0
  - Labels:       "SAFE" (benign) | "INJECTION" (prompt injection detected)
  - Max tokens:   512 (truncation enforced)
  - Limitation:   English-only, does not detect jailbreaks (use DeterministicStage
                  bypass_patterns for those)

References:
  ProtectAI (2024). Fine-Tuned DeBERTa-v3-base for Prompt Injection Detection.
  https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection-v2

  Greshake et al. (2023). Not what you've signed up for: Compromising real-world
  LLM-integrated applications with indirect prompt injection.
  https://doi.org/10.1145/3605764.3623985

  OWASP (2024). OWASP Top 10 for Large Language Model Applications — LLM01: Prompt Injection.
  https://owasp.org/www-project-top-10-for-large-language-model-applications/
"""

import logging
from typing import Any

from src.models import Label
from src.stages.semantic import SemanticBERTStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label map specific to ProtectAI/deberta-v3-base-prompt-injection-v2.
#
# The model outputs exactly two labels:
#   "SAFE"      → benign input, no injection detected
#   "INJECTION" → prompt injection attack detected
#
# "INJECTION" maps to Label.MALIGN (not Crisis) because prompt injection
# is a security/integrity threat, not a patient safety crisis.
# This distinction matters for the downstream response code:
#   Maligna → 400, Crisis → 406
# ---------------------------------------------------------------------------
_ATTACK_LABEL_MAP: dict[str, Label] = {
    "SAFE": Label.VALID,
    "INJECTION": Label.MALIGN,
}


class AttackDetectionStage(SemanticBERTStage):
    """
    Semantic stage specialised for LLM prompt-injection attack detection.

    Inherits all infrastructure from SemanticBERTStage:
      - Lazy loading with double-checked locking (thread-safe)
      - Configurable inference timeout with fail-closed on timeout
      - Injectable pipeline_factory for testing without GPU
      - Threshold-based confidence gating

    What this class overrides:
      - _NAME: distinct stage name for logging and metadatos
      - default model_id: ProtectAI/deberta-v3-base-prompt-injection-v2
      - default label_map: SAFE → Válida, INJECTION → Maligna

    The DeterministicStage already covers known bypass patterns via regex.
    This stage adds semantic coverage for novel or paraphrased injections
    that evade keyword/regex matching.

    Operational note: this model does NOT detect jailbreaks and is English-only.
    For multilingual support, consider replacing with
    ProtectAI/deberta-v3-base-prompt-injection-v2 fine-tuned on multilingual data
    or meta-llama/Prompt-Guard-86M (requires Meta license acceptance).
    """

    _NAME = "attack_detection"

    _DEFAULT_MODEL_ID = "ProtectAI/deberta-v3-base-prompt-injection-v2"

    def __init__(
            self,
            model_id: str = _DEFAULT_MODEL_ID,
            threshold: float = 0.85,  # higher default: prefer precision over recall for attacks
            inference_timeout_s: int = 10,
            pipeline_factory: Any = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            threshold=threshold,
            inference_timeout_s=inference_timeout_s,
            label_map=_ATTACK_LABEL_MAP,
            pipeline_factory=pipeline_factory,
        )
        logger.info(
            "AttackDetectionStage configured — model=%s threshold=%.2f",
            model_id,
            threshold,
        )

    @property
    def name(self) -> str:
        return self._NAME
