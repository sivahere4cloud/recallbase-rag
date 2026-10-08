import logging

from recallbase_rag.schemas import Chunk, Page

logger = logging.getLogger(__name__)

BREAK_SEPARATORS = ("\n\n", "\n", " ")


def chunk_pages(
    pages: list[Page],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be 0 or more and smaller than chunk_size")

    chunks: list[Chunk] = []
    chunk_index = 0

    for page in pages:
        pieces = split_text(page.text, chunk_size, chunk_overlap)
        for piece in pieces:
            chunk = Chunk(
                chunk_id=f"{page.file_name}-p{page.page_number}-c{chunk_index}",
                file_name=page.file_name,
                page_number=page.page_number,
                chunk_index=chunk_index,
                text=piece,
            )
            chunks.append(chunk)
            chunk_index += 1

    logger.info("Made %s chunks from %s pages", len(chunks), len(pages))
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    pieces: list[str] = []
    length = len(text)
    start = 0

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = find_break(text, start, end)

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= length:
            break

        next_start = max(end - chunk_overlap, start + 1)
        start = move_to_word_start(text, next_start)

    return pieces


def find_break(text: str, start: int, end: int) -> int:
    lower_limit = start + (end - start) // 2

    for separator in BREAK_SEPARATORS:
        position = text.rfind(separator, lower_limit, end)
        if position != -1:
            return position

    return end


def move_to_word_start(text: str, start: int) -> int:
    length = len(text)

    if start > 0 and not text[start - 1].isspace():
        while start < length and not text[start].isspace():
            start += 1

    while start < length and text[start].isspace():
        start += 1

    return start
    