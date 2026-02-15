from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings

# PATH
RESOURCES = Path(__file__).parent.parent.parent / "resources"


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # HuggingFace credentials
    # Loaded from HF_TOKEN in .env or environment.
    # SecretStr prevents the token from appearing in logs or repr().
    # Required when enable_semantic_stage=True.
    # ------------------------------------------------------------------
    hf_token: SecretStr | None = Field(
        default=None,
        alias="HF_TOKEN",
    )

    # ------------------------------------------------------------------
    # Semantic stage
    # ------------------------------------------------------------------
    model_id: str = Field(
        default="mental/mental-bert-base-uncased",
        alias="SAFEGUARD_MODEL_ID",
    )
    model_threshold: float = Field(
        default=0.75,
        alias="SAFEGUARD_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    inference_timeout_s: int = Field(
        default=10,
        alias="SAFEGUARD_TIMEOUT",
        gt=0,
    )

    # ------------------------------------------------------------------
    # Resource files (keywords / patterns)
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

    # ------------------------------------------------------------------
    # Pipeline feature flags
    # ------------------------------------------------------------------
    enable_semantic_stage: bool = Field(
        default=True,
        alias="SAFEGUARD_ENABLE_BERT",
    )

    model_config = {
        "env_file": ".env",
        "populate_by_name": True,
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def require_token_when_bert_enabled(self) -> "Settings":
        """
        Fail fast at Settings construction time if BERT is enabled but
        no HF token is provided. This prevents a silent startup where
        the model load would fail later with a cryptic auth error.
        """
        if self.enable_semantic_stage and not self.hf_token:
            raise ValueError(
                "HF_TOKEN is required when SAFEGUARD_ENABLE_BERT=true. "
                "Set it in your .env file or environment. "
                "To run without BERT, set SAFEGUARD_ENABLE_BERT=false."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Use dependency injection in FastAPI: Depends(get_settings).
    Call get_settings.cache_clear() in tests to reset state.
    """
    return Settings()
