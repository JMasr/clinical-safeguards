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
    Also raised for invalid pipeline configuration (e.g. unknown stage targets).

    `cause` is optional — omit it for configuration errors where there is no
    underlying exception (e.g. registry validation failures).
    """

    def __init__(self, resource: str, cause: Exception | None = None) -> None:
        self.resource = resource
        self.cause = cause
        if cause is not None:
            msg = f"Failed to load resource '{resource}': {type(cause).__name__}: {cause}"
        else:
            msg = f"Resource error: {resource}"
        super().__init__(msg)
