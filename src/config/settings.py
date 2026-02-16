from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

# PATH
RESOURCES = Path(__file__).parent.parent.parent / "resources"

class Settings(BaseSettings):
    """
    Runtime settings loaded from environment / .env file.

    Responsibility scope (post-Hydra refactor):
      - Secrets that must NOT appear in config files (HF_TOKEN)
      - Infrastructure settings shared across all stages (timeout)
      - Resource file paths (can be overridden per-environment)

    Pipeline composition (which stages, in which order, with which
    model IDs and thresholds) is now fully owned by Hydra configs
    under conf/pipeline/*.yaml — not here.
    """

    # ------------------------------------------------------------------
    # HuggingFace credentials
    # Loaded from HF_TOKEN in .env or environment.
    # SecretStr prevents the token from appearing in logs or repr().
    # ------------------------------------------------------------------
    hf_token: SecretStr | None = Field(
        default=None,
        alias="HF_TOKEN",
    )

    # ------------------------------------------------------------------
    # Shared inference settings — apply to all semantic stages unless
    # overridden per-stage in the Hydra config.
    # ------------------------------------------------------------------
    inference_timeout_s: int = Field(
        default=10,
        alias="SAFEGUARD_TIMEOUT",
        gt=0,
    )

    # ------------------------------------------------------------------
    # Resource files (keywords / patterns)
    # Hydra configs reference these via ${paths.*} interpolation.
    # Override per-environment via env var or Hydra CLI override.
    # ------------------------------------------------------------------
    keywords_crisis_path: Path = Field(
        default=Path("resources/keywords_crisis.yaml"),
        alias="SAFEGUARD_CRISIS_KEYWORDS",
    )
    keywords_malign_path: Path = Field(
        default=Path("resources/keywords_malign.yaml"),
        alias="SAFEGUARD_MALIGN_KEYWORDS",
    )
    bypass_patterns_path: Path = Field(
        default=Path("resources/bypass_patterns.yaml"),
        alias="SAFEGUARD_BYPASS_PATTERNS",
    )

    model_config = {
        "env_file": ".env",
        "populate_by_name": True,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()