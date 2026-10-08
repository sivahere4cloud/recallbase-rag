import pytest

from recallbase_rag.chunker import chunk_pages
from recallbase_rag.cleaner import clean_pages, clean_text
from recallbase_rag.schemas import Page


def test_clean_text_joins_hyphenated_line_breaks() -> None:
    assert clean_text("exam-\nple text") == "example text"


def test_clean_text_removes_soft_hyphen() -> None:
    assert clean_text("co\u00adoperate") == "cooperate"


def test_clean_text_squeezes_spaces() -> None:
    assert clean_text("a      b") == "a b"


def test_clean_text_limits_blank_lines() -> None:
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_text_replaces_private_use_characters() -> None:
    assert clean_text("a\uf03cb") == "a b"


def test_clean_pages_keeps_file_name_and_page_number() -> None:
    pages = [Page(file_name="f.txt", page_number=3, text="  hello   world  ")]

    cleaned = clean_pages(pages)

    assert cleaned[0].file_name == "f.txt"
    assert cleaned[0].page_number == 3
    assert cleaned[0].text == "hello world"


def test_chunk_pages_rejects_overlap_not_smaller_than_size() -> None:
    pages = [Page(file_name="f.txt", page_number=1, text="hello")]

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=50, chunk_overlap=50)


def test_short_text_makes_one_chunk_with_expected_id() -> None:
    pages = [Page(file_name="f.txt", page_number=1, text="hello world")]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "f.txt-p1-c0"
    assert chunks[0].text == "hello world"


def test_chunks_respect_size_and_never_cut_words() -> None:
    text = "word " * 100
    pages = [Page(file_name="f.txt", page_number=1, text=text.strip())]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 100
        for word in chunk.text.split():
            assert word == "word"


def test_chunks_overlap_and_cover_every_word() -> None:
    words: list[str] = []
    for number in range(60):
        words.append(f"w{number}")
    pages = [Page(file_name="f.txt", page_number=1, text=" ".join(words))]

    chunks = chunk_pages(pages, chunk_size=60, chunk_overlap=20)

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert second_words[0] in first_words

    seen: set[str] = set()
    for chunk in chunks:
        for word in chunk.text.split():
            seen.add(word)
    assert seen == set(words)


def test_chunks_keep_page_numbers_and_global_index() -> None:
    pages = [
        Page(file_name="f.txt", page_number=1, text="first page text"),
        Page(file_name="f.txt", page_number=2, text="second page text"),
    ]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=10)

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[1].chunk_id == "f.txt-p2-c1"