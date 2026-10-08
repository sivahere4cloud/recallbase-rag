class RecallbaseError(Exception):
    """Base class for all errors raised by this project."""


class DocumentLoadError(RecallbaseError):
    """A file could not be found, opened or read."""


class EmbeddingError(RecallbaseError):
    """The embedding model failed to turn text into vectors."""


class VectorStoreError(RecallbaseError):
    """The vector database failed to save or search."""


class LLMError(RecallbaseError):
    """The language model call failed."""