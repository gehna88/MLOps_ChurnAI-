"""Project-specific exception hierarchy.

A single base class lets callers (CLI entry points, the API) catch every
expected failure in one place while still letting genuine bugs surface.
"""


class ChurnAIError(Exception):
    """Base class for all expected ChurnAI failures."""


class ConfigError(ChurnAIError):
    """Configuration file is missing or invalid."""


class DataValidationError(ChurnAIError):
    """Input data violates the data contract."""


class LeakageSuspectedError(ChurnAIError):
    """Model performance is implausibly high, suggesting target leakage."""


class ModelNotLoadedError(ChurnAIError):
    """A prediction was requested before a model artifact was available."""
