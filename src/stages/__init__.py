from __future__ import annotations

from .attack_detection import AttackDetectionStage
from .deterministic import DeterministicStage
from .semantic import SemanticBERTStage

# ---------------------------------------------------------------------------
# STAGE_REGISTRY — single source of truth for valid pipeline stages.
#
# Every class listed here can be referenced via _target_ in Hydra configs.
# _build_pipeline() validates all _target_ values against this set before
# calling hydra.utils.instantiate() — prevents arbitrary code execution via
# a malformed config file.
#
# Adding a new stage:
#   1. Implement the class (inheriting GuardrailStage)
#   2. Import it above
#   3. Add its fully-qualified name to STAGE_REGISTRY
#   No other file needs to change.
# ---------------------------------------------------------------------------
STAGE_REGISTRY: frozenset[str] = frozenset(
    {
        "src.stages.deterministic.DeterministicStage",
        "src.stages.semantic.SemanticBERTStage",
        "src.stages.attack_detection.AttackDetectionStage",
    }
)

__all__ = [
    "AttackDetectionStage",
    "DeterministicStage",
    "SemanticBERTStage",
    "STAGE_REGISTRY",
]
