import re
import zlib

from recallbase_rag.prompt_builder import Prompt
from recallbase_rag.schemas import Chunk, SearchHit

WORD_PATTERN = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Turns words into a small vector. Same words give similar vectors."""

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(self.embed_text(text))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self.embed_text(text)

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension

        for word in WORD_PATTERN.findall(text.lower()):
            position = zlib.crc32(word.encode("utf-8")) % self.dimension
            vector[position] += 1.0

        length = sum(value * value for value in vector) ** 0.5
        if length == 0:
            return vector

        normalised: list[float] = []
        for value in vector:
            normalised.append(value / length)
        return normalised


class FakeVectorStore:
    """Keeps chunks in a list. Search ranks by dot product."""

    def __init__(self) -> None:
        self.items: list[tuple[Chunk, list[float]]] = []

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        for chunk, vector in zip(chunks, vectors, strict=True):
            self.items.append((chunk, vector))

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        hits: list[SearchHit] = []

        for chunk, stored_vector in self.items:
            score = 0.0
            for left, right in zip(vector, stored_vector, strict=True):
                score += left * right
            hits.append(SearchHit(chunk=chunk, score=score))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def delete_document(self, file_name: str) -> None:
        kept: list[tuple[Chunk, list[float]]] = []
        for chunk, vector in self.items:
            if chunk.file_name != file_name:
                kept.append((chunk, vector))
        self.items = kept


class FakeLLM:
    """Returns a fixed answer and remembers the prompt it was given."""

    def __init__(self, answer: str = "fake answer") -> None:
        self.answer = answer
        self.prompts: list[Prompt] = []

    def generate(self, prompt: Prompt) -> str:
        self.prompts.append(prompt)
        return self.answer