"""Gradio page: upload documents, ask questions, see sources.

This file stays thin. All real work happens in RagService.
"""

import logging
from pathlib import Path

import gradio as gr

from recallbase_rag.bootstrap import get_service
from recallbase_rag.errors import RecallbaseError
from recallbase_rag.schemas import Answer

logger = logging.getLogger(__name__)

ALLOWED_TYPES = [".pdf", ".txt", ".md"]


def ingest_files(file_paths: list[str] | None) -> str:
    """Store every uploaded file. Return a status message."""
    if not file_paths:
        return "Choose at least one file first."

    service = get_service()
    lines: list[str] = []

    for file_path in file_paths:
        path = Path(file_path)
        try:
            count = service.store_document(path)
            lines.append(f"OK: {path.name} stored as {count} chunks.")
        except RecallbaseError as error:
            logger.warning("Could not store %s: %s", path.name, error)
            lines.append(f"FAILED: {path.name}: {error}")

    return "\n".join(lines)


def format_sources(answer: Answer) -> str:
    """Turn the sources into a small markdown list."""
    if not answer.sources:
        return "No sources."

    lines: list[str] = []
    number = 1
    for hit in answer.sources:
        chunk = hit.chunk
        line = f"{number}. **{chunk.file_name}**, page {chunk.page_number} (score {hit.score:.3f})"
        lines.append(line)
        number += 1

    return "\n".join(lines)


def ask(question: str) -> tuple[str, str]:
    """Answer one question. Return (answer text, sources text)."""
    clean_question = question.strip()
    if not clean_question:
        return "Type a question first.", ""

    try:
        answer = get_service().ask_question(clean_question)
    except RecallbaseError as error:
        logger.warning("Question failed: %s", error)
        return f"Error: {error}", ""

    return answer.text, format_sources(answer)


def build_app() -> gr.Blocks:
    """Build the page. Nothing runs until launch()."""
    with gr.Blocks(title="recallbase-rag") as app:
        gr.Markdown("# recallbase-rag\nUpload documents, then ask questions about them.")

        with gr.Row():
            with gr.Column():
                files = gr.File(
                    label="Documents (PDF, TXT, MD)",
                    file_count="multiple",
                    file_types=ALLOWED_TYPES,
                    type="filepath",
                )
                store_button = gr.Button("Store documents")
                status = gr.Textbox(label="Status", lines=4, interactive=False)

            with gr.Column():
                question = gr.Textbox(label="Question", lines=2)
                ask_button = gr.Button("Ask", variant="primary")
                answer_box = gr.Markdown(label="Answer")
                sources_box = gr.Markdown(label="Sources")

        store_button.click(
            ingest_files,
            inputs=files,
            outputs=status,
            concurrency_limit=1,
        )
        ask_button.click(
            ask,
            inputs=question,
            outputs=[answer_box, sources_box],
            concurrency_limit=1,
        )
        question.submit(
            ask,
            inputs=question,
            outputs=[answer_box, sources_box],
            concurrency_limit=1,
        )

    return app


def main(host: str = "127.0.0.1", port: int = 7860) -> None:
    build_app().launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()