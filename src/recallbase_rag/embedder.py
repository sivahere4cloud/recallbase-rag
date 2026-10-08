import logging
from typing import Protocol

from sentence_transformers import SentenceTransformer

from recallbase_rag.errors import EmbeddingError

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        dimension: int,
        query_prefix: str = BGE_QUERY_PREFIX,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.query_prefix = query_prefix
        self.batch_size = batch_size

        logger.info("Loading embedding model %s", model_name)
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as error:
            raise EmbeddingError(
                f"Could not load embedding model {model_name}: {error}"
            ) from error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encode_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self.encode_texts([self.query_prefix + text])
        return vectors[0]

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise EmbeddingError(f"Embedding failed: {error}") from error

        vectors = embeddings.tolist()

        if len(vectors[0]) != self.dimension:
            raise EmbeddingError(
                f"Expected vectors of size {self.dimension}, "
                f"but {self.model_name} gave size {len(vectors[0])}"
            )

        return vectors
        