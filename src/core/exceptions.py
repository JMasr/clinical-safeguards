from __future__ import annotations


class SafeguardError(Exception):
    """
    Base for all internal errors.
    Any unhandled SafeguardError in the pipeline → fail-closed (500).
    """


class StageExecutionError(SafeguardError):
    """
    Raised when a stage fails during process() — timeout, OOM, network, etc.
    Wraps the original exception to preserve the full traceback.
    """

    def __init__(self, stage_name: str, cause: Exception) -> None:
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(
            f"Stage '{stage_name}' failed: {type(cause).__name__}: {cause}"
        )


class ResourceLoadError(SafeguardError):
    """
    Raised when a required resource cannot be loaded at startup:
    keyword YAML files, bypass pattern files, model weights, etc.
    """

    def __init__(self, resource: str, cause: Exception) -> None:
        self.resource = resource
        self.cause = cause
        super().__init__(
            f"Failed to load resource '{resource}': {type(cause).__name__}: {cause}"
        )
