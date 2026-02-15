from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from clinical_safeguard.config import initialize_hf_services
from clinical_safeguard.core.exceptions import ResourceLoadError


def _token(value: str = "hf_testtoken") -> SecretStr:
    return SecretStr(value)


class TestInitializeHFServices:
    def test_calls_huggingface_hub_login(self) -> None:
        mock_hub = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            initialize_hf_services(_token("hf_abc123"))
        mock_hub.login.assert_called_once_with(
            token="hf_abc123",
            add_to_git_credential=False,
        )

    def test_token_value_passed_correctly(self) -> None:
        mock_hub = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            initialize_hf_services(_token("hf_supersecret"))
        call_kwargs = mock_hub.login.call_args.kwargs
        assert call_kwargs["token"] == "hf_supersecret"

    def test_git_credential_is_never_set(self) -> None:
        """Token must never be written to git credentials."""
        mock_hub = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            initialize_hf_services(_token())
        assert mock_hub.login.call_args.kwargs["add_to_git_credential"] is False

    def test_missing_huggingface_hub_raises_resource_load_error(self) -> None:
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ResourceLoadError) as exc_info:
                initialize_hf_services(_token())
        assert "huggingface_hub" in exc_info.value.resource
        assert isinstance(exc_info.value.cause, ImportError)

    def test_login_failure_raises_resource_load_error(self) -> None:
        mock_hub = MagicMock()
        mock_hub.login.side_effect = ValueError("Invalid token")
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            with pytest.raises(ResourceLoadError) as exc_info:
                initialize_hf_services(_token("hf_bad"))
        assert "huggingface_hub.login" in exc_info.value.resource
        assert isinstance(exc_info.value.cause, ValueError)

    def test_network_error_raises_resource_load_error(self) -> None:
        mock_hub = MagicMock()
        mock_hub.login.side_effect = OSError("Connection refused")
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            with pytest.raises(ResourceLoadError):
                initialize_hf_services(_token())
