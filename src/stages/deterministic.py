from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

import yaml
from flashtext import KeywordProcessor

from src.core.base import GuardrailStage
from src.core.exceptions import ResourceLoadError, StageExecutionError
from src.models import Label, PromptInput, StageResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

class _BypassPattern(NamedTuple):
    pattern_id: str
    description: str
    compiled: re.Pattern


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class DeterministicStage(GuardrailStage):
    """
    Phase 1: deterministic keyword and regex matching.

    Decision logic (in order of priority):
      1. Bypass patterns (regex) → Maligna, short_circuit=True
      2. Crisis keywords          → Crisis,  short_circuit=True
      3. Malign keywords          → Maligna, short_circuit=True
      4. No match                 → Válida,  short_circuit=False

    Crisis takes precedence over Malign in keyword matching because a prompt
    that contains both signals should always route to the crisis protocol.

    Resources are loaded once at construction time. Any failure raises
    ResourceLoadError so the service startup fails fast rather than silently
    operating in a degraded state.
    """

    _NAME = "deterministic"

    def __init__(
            self,
            keywords_crisis_path: Path,
            keywords_malign_path: Path,
            bypass_patterns_path: Path,
    ) -> None:
        try:
            self._crisis_processor = self._build_keyword_processor(
                keywords_crisis_path, label="crisis"
            )
            self._malign_processor = self._build_keyword_processor(
                keywords_malign_path, label="malign"
            )
            self._bypass_patterns: list[_BypassPattern] = self._load_bypass_patterns(
                bypass_patterns_path
            )
        except ResourceLoadError:
            raise
        except Exception as exc:
            raise ResourceLoadError("DeterministicStage initialization", exc) from exc

        logger.info(
            "DeterministicStage loaded — crisis_keywords=%d, malign_keywords=%d, bypass_patterns=%d",
            len(self._crisis_processor),
            len(self._malign_processor),
            len(self._bypass_patterns),
        )

    # ------------------------------------------------------------------
    # GuardrailStage interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._NAME

    def process(self, prompt: PromptInput) -> StageResult:
        try:
            return self._evaluate(prompt)
        except StageExecutionError:
            raise
        except Exception as exc:
            raise StageExecutionError(self._NAME, exc) from exc

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, prompt: PromptInput) -> StageResult:
        text_lower = prompt.text.lower()

        # Priority 1: bypass / prompt-injection patterns
        bypass_hit = self._check_bypass(text_lower)
        if bypass_hit:
            logger.warning("Bypass pattern matched: %s", bypass_hit)
            return StageResult(
                stage_name=self._NAME,
                label=Label.MALIGN,
                confidence=1.0,
                triggered_by=f"bypass:{bypass_hit}",
                short_circuit=True,
            )

        # Priority 2: crisis keywords (C-SSRS aligned)
        crisis_hit = self._crisis_processor.extract_keywords(text_lower, span_info=False)
        if crisis_hit:
            keyword = crisis_hit[0]
            logger.warning("Crisis keyword matched: '%s'", keyword)
            return StageResult(
                stage_name=self._NAME,
                label=Label.CRISIS,
                confidence=1.0,
                triggered_by=f"keyword:{keyword}",
                short_circuit=True,
            )

        # Priority 3: malign keywords
        malign_hit = self._malign_processor.extract_keywords(text_lower, span_info=False)
        if malign_hit:
            keyword = malign_hit[0]
            logger.warning("Malign keyword matched: '%s'", keyword)
            return StageResult(
                stage_name=self._NAME,
                label=Label.MALIGN,
                confidence=1.0,
                triggered_by=f"keyword:{keyword}",
                short_circuit=True,
            )

        # No match — pass to next stage
        return StageResult(
            stage_name=self._NAME,
            label=Label.VALID,
            confidence=1.0,
            triggered_by=None,
            short_circuit=False,
        )

    def _check_bypass(self, text_lower: str) -> str | None:
        """Return the pattern_id of the first matching bypass pattern, or None."""
        for bp in self._bypass_patterns:
            if bp.compiled.search(text_lower):
                return bp.pattern_id
        return None

    # ------------------------------------------------------------------
    # Resource loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_keyword_processor(path: Path, label: str) -> KeywordProcessor:
        """
        Load keywords from a YAML file into a FlashText KeywordProcessor.

        FlashText uses the Aho-Corasick algorithm (O(n) per text regardless of
        keyword count), making it significantly faster than iterating regex
        patterns for large keyword banks.
        """
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, yaml.YAMLError) as exc:
            raise ResourceLoadError(str(path), exc) from exc

        processor = KeywordProcessor(case_sensitive=False)
        categories = raw.get("categories", {})

        for category_name, category_data in categories.items():
            keywords = category_data.get("keywords", [])
            for kw in keywords:
                # Map keyword → canonical keyword string (used as triggered_by)
                processor.add_keyword(kw)

        if len(processor) == 0:
            logger.warning(
                "Keyword processor for '%s' loaded 0 keywords from '%s'. "
                "Check the YAML structure.",
                label,
                path,
            )

        return processor

    @staticmethod
    def _load_bypass_patterns(path: Path) -> list[_BypassPattern]:
        """
        Load and compile regex bypass patterns from YAML.
        Patterns are compiled once at startup — not per request.
        """
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, yaml.YAMLError) as exc:
            raise ResourceLoadError(str(path), exc) from exc

        patterns: list[_BypassPattern] = []
        for entry in raw.get("patterns", []):
            try:
                compiled = re.compile(entry["pattern"], re.IGNORECASE | re.UNICODE)
                patterns.append(
                    _BypassPattern(
                        pattern_id=entry["id"],
                        description=entry.get("description", ""),
                        compiled=compiled,
                    )
                )
            except re.error as exc:
                raise ResourceLoadError(
                    f"bypass pattern id={entry.get('id', '?')}", exc
                ) from exc

        return patterns
