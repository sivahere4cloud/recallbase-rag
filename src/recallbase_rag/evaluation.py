import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from recallbase_rag.errors import EvaluationError
from recallbase_rag.retriever import Retriever

logger = logging.getLogger(__name__)

CUTOFFS = (1, 3, 5, 8)
MAX_QUESTION_WIDTH = 70


class EvalQuestion(BaseModel):
    question: str = Field(min_length=1)
    pages: list[int] = Field(min_length=1)


@dataclass(frozen=True)
class QuestionResult:
    question: str
    expected_pages: list[int]
    retrieved_pages: list[int]
    rank: int | None


@dataclass(frozen=True)
class EvalReport:
    top_k: int
    results: list[QuestionResult]

    def hit_rate(self, n: int) -> float:
        if not self.results:
            return 0.0

        hits = 0
        for result in self.results:
            if result.rank is not None and result.rank <= n:
                hits += 1

        return hits / len(self.results)

    def mean_reciprocal_rank(self) -> float:
        if not self.results:
            return 0.0

        total = 0.0
        for result in self.results:
            if result.rank is not None:
                total += 1.0 / result.rank

        return total / len(self.results)


def load_questions(path: Path) -> list[EvalQuestion]:
    if not path.is_file():
        raise EvaluationError(f"Questions file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Could not read {path}: {error}") from error

    if not isinstance(raw, list) or not raw:
        raise EvaluationError("The questions file must hold a non-empty JSON list.")

    questions: list[EvalQuestion] = []
    for item in raw:
        try:
            questions.append(EvalQuestion.model_validate(item))
        except ValidationError as error:
            raise EvaluationError(f"Bad question entry {item!r}: {error}") from error

    return questions


def evaluate(questions: list[EvalQuestion], retriever: Retriever) -> EvalReport:
    results: list[QuestionResult] = []

    for item in questions:
        hits = retriever.retrieve(item.question)

        retrieved_pages: list[int] = []
        for hit in hits:
            retrieved_pages.append(hit.chunk.page_number)

        rank: int | None = None
        for position, page in enumerate(retrieved_pages, start=1):
            if page in item.pages:
                rank = position
                break

        results.append(
            QuestionResult(
                question=item.question,
                expected_pages=item.pages,
                retrieved_pages=retrieved_pages,
                rank=rank,
            )
        )

    logger.info("Evaluated %s questions", len(results))
    return EvalReport(top_k=retriever.top_k, results=results)


def format_report(report: EvalReport) -> str:
    cutoffs: list[int] = []
    for n in CUTOFFS:
        if n < report.top_k:
            cutoffs.append(n)
    cutoffs.append(report.top_k)

    summary_parts: list[str] = []
    for n in cutoffs:
        summary_parts.append(f"Hit@{n}: {report.hit_rate(n):.0%}")
    summary_parts.append(f"MRR: {report.mean_reciprocal_rank():.2f}")

    lines: list[str] = []
    lines.append(f"Questions: {len(report.results)}   top_k: {report.top_k}")
    lines.append("   ".join(summary_parts))
    lines.append("")

    for result in report.results:
        rank_text = "MISS" if result.rank is None else f"rank {result.rank}"
        question_text = result.question
        if len(question_text) > MAX_QUESTION_WIDTH:
            question_text = question_text[: MAX_QUESTION_WIDTH - 3] + "..."
        lines.append(
            f"{rank_text:>7} | {question_text} | "
            f"expected {result.expected_pages} | got {result.retrieved_pages}"
        )

    return "\n".join(lines)