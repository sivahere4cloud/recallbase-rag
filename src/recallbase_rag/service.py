import logging
import time
from pathlib import Path

from recallbase_rag.chunker import chunk_pages
from recallbase_rag.cleaner import clean_pages
from recallbase_rag.embedder import Embedder
from recallbase_rag.errors import DocumentLoadError
from recallbase_rag.llm import LLM
from recallbase_rag.loader import load_document
from recallbase_rag.prompt_builder import build_prompt
from recallbase_rag.retriever import Retriever
from recallbase_rag.schemas import Answer
from recallbase_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

NO_RESULTS_TEXT = (
    "I could not find anything in the stored documents. Load a document first."
)


class RagService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever = Retriever(embedder, store, top_k)

    def store_document(self, path: Path) -> int:
        started = time.perf_counter()

        pages = load_document(path)
        cleaned_pages = clean_pages(pages)
        chunks = chunk_pages(cleaned_pages, self.chunk_size, self.chunk_overlap)

        if not chunks:
            raise DocumentLoadError(f"No text left in {path.name} after cleaning.")

        texts: list[str] = []
        for chunk in chunks:
            texts.append(chunk.text)
        vectors = self.embedder.embed_documents(texts)

        self.store.delete_document(path.name)
        self.store.add_chunks(chunks, vectors)

        elapsed = time.perf_counter() - started
        logger.info(
            "Stored %s: %s pages, %s chunks in %.1f seconds",
            path.name,
            len(pages),
            len(chunks),
            elapsed,
        )
        return len(chunks)

    def ask_question(self, question: str) -> Answer:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question must not be empty")

        started = time.perf_counter()
        hits = self.retriever.retrieve(cleaned_question)
        retrieval_seconds = time.perf_counter() - started

        if not hits:
            return Answer(
                question=cleaned_question,
                text=NO_RESULTS_TEXT,
                sources=[],
            )

        prompt = build_prompt(cleaned_question, hits)
        answer_text = self.llm.generate(prompt)

        total_seconds = time.perf_counter() - started
        logger.info(
            "Answered question: retrieval %.2f seconds, total %.2f seconds",
            retrieval_seconds,
            total_seconds,
        )

        return Answer(
            question=cleaned_question,
            text=answer_text,
            sources=hits,
        )