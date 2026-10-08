import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from recallbase_rag.errors import DocumentLoadError
from recallbase_rag.schemas import Page

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md"}


def load_document(path: Path) -> list[Page]:
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)

    if suffix in TEXT_SUFFIXES:
        return load_text_file(path)

    raise DocumentLoadError(f"Unsupported file type: {path.suffix}")


def load_pdf(path: Path) -> list[Page]:
    pages: list[Page] = []
    empty_pages = 0

    try:
        reader = PdfReader(path)
        for index, pdf_page in enumerate(reader.pages):
            text = pdf_page.extract_text()
            if not text.strip():
                empty_pages += 1
            page = Page(
                file_name=path.name,
                page_number=index + 1,
                text=text,
            )
            pages.append(page)
    except PdfReadError as error:
        raise DocumentLoadError(f"Could not read PDF {path.name}: {error}") from error

    if empty_pages == len(pages):
        raise DocumentLoadError(
            f"No text found in {path.name}. It may be a scanned PDF."
        )

    if empty_pages > 0:
        logger.warning("%s: %s pages had no text", path.name, empty_pages)

    logger.info("Loaded %s pages from %s", len(pages), path.name)
    return pages


def load_text_file(path: Path) -> list[Page]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DocumentLoadError(f"Could not read {path.name}: {error}") from error

    if not text.strip():
        raise DocumentLoadError(f"File is empty: {path.name}")

    page = Page(file_name=path.name, page_number=1, text=text)
    logger.info("Loaded 1 page from %s", path.name)
    return [page]