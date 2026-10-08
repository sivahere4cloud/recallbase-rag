import logging

from recallbase_rag.embedder import Embedder
from recallbase_rag.schemas import SearchHit
from recallbase_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[SearchHit]:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question must not be empty")

        query_vector = self.embedder.embed_query(cleaned_question)
        hits = self.store.search(query_vector, self.top_k)

        if hits:
            logger.info(
                "Retrieved %s chunks, best score %.3f",
                len(hits),
                hits[0].score,
            )
        else:
            logger.warning("No chunks found for the question")

        return hits