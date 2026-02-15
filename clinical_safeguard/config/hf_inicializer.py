from __future__ import annotations

import logging

from pydantic import SecretStr

logger = logging.getLogger(__name__)


def initialize_hf_services(hf_token: SecretStr) -> None:
    """
    Authenticate with HuggingFace Hub and propagate the token to all
    HF libraries that respect the hub login state.

    Calling huggingface_hub.login() is the single canonical entry point:
    - transformers.pipeline() and AutoModel.from_pretrained() read the
      cached token automatically via huggingface_hub internals.
    - datasets.load_dataset() does the same.
    No per-library configuration is required.

    Must be called BEFORE any model or dataset is loaded.
    Raises ResourceLoadError if the huggingface_hub library is not installed
    or if the login call itself fails (invalid token, network error).

    References:
      https://huggingface.co/docs/huggingface_hub/en/guides/cli#login
      https://huggingface.co/docs/transformers/en/installation#using-a-specific-token
    """
    from clinical_safeguard.core.exceptions import ResourceLoadError  # noqa: PLC0415

    try:
        import huggingface_hub  # noqa: PLC0415
    except ImportError as exc:
        raise ResourceLoadError(
            "huggingface_hub",
            ImportError(
                "huggingface_hub is not installed. "
                "Run: pip install huggingface_hub"
            ),
        ) from exc

    token_value = hf_token.get_secret_value()

    try:
        huggingface_hub.login(token=token_value, add_to_git_credential=False)
        logger.info(
            "HuggingFace Hub authenticated successfully. "
            "Token propagated to transformers and datasets."
        )
    except Exception as exc:
        raise ResourceLoadError(
            "huggingface_hub.login",
            exc,
        ) from exc
