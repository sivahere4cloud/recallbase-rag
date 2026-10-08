import argparse
import sys
from pathlib import Path

from recallbase_rag.bootstrap import get_service
from recallbase_rag.errors import RecallbaseError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recallbase-rag",
        description="Ask questions about your documents and see the sources.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load a document into the vector store",
    )
    ingest_parser.add_argument(
        "path",
        type=Path,
        help="Path to a .pdf, .txt or .md file",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question about the stored documents",
    )
    ask_parser.add_argument("question", help="Your question, in quotes")

    search_parser = subparsers.add_parser(
        "search",
        help="Show the retrieved chunks for a question (no OpenAI call)",
    )
    search_parser.add_argument("question", help="Your question, in quotes")
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="How many chunks to show (default: from settings)",
    )

    return parser


def run_ingest(path: Path) -> None:
    service = get_service()
    chunk_count = service.store_document(path)
    print(f"Stored {chunk_count} chunks from {path.name}")


def run_ask(question: str) -> None:
    service = get_service()
    answer = service.ask_question(question)

    print()
    print(answer.text)
    print()

    if answer.sources:
        print("Sources:")
        for number, hit in enumerate(answer.sources, start=1):
            chunk = hit.chunk
            print(
                f"[{number}] {chunk.file_name}, page {chunk.page_number} "
                f"(score {hit.score:.3f})"
            )


def run_search(question: str, top_k: int | None) -> None:
    service = get_service()
    if top_k is not None:
        service.retriever.top_k = top_k

    hits = service.retriever.retrieve(question)
    if not hits:
        print("No chunks found. Did you run ingest?")
        return

    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        print()
        print(
            f"[{number}] score {hit.score:.3f} | {chunk.file_name} | "
            f"page {chunk.page_number} | chunk {chunk.chunk_index}"
        )
        print(chunk.text)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "ingest":
            run_ingest(args.path)
        elif args.command == "ask":
            run_ask(args.question)
        elif args.command == "search":
            run_search(args.question, args.top_k)
    except (RecallbaseError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())