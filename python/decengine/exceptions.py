"""Typed exceptions mapped from the stable native status codes."""


class DecengineError(Exception):
    """Base class for all SDK errors."""


class NativeLibraryNotFound(DecengineError):
    """The separately installed native library could not be resolved."""


class InvalidRequest(DecengineError):
    """The decision request is malformed."""


class InvalidQuestion(InvalidRequest):
    """A question fails public API validation."""


class ModelNotFound(DecengineError):
    """An allowlisted model has not been installed."""


class UnsupportedModel(DecengineError):
    """The requested model is not a v1 compatibility target."""


class ModelLoadFailed(DecengineError):
    """An installed model could not be loaded or verified."""


class ContextTooLong(DecengineError):
    """Tokenized input exceeds the model profile limit."""


class OutOfMemory(DecengineError):
    """The native engine ran out of unified memory."""


class InvalidManifest(DecengineError):
    """A model manifest is invalid or incompatible."""


class EngineError(DecengineError):
    """The native decision engine failed."""
