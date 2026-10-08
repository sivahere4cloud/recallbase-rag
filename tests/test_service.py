from pathlib import Path

import pytest
from fakes import FakeEmbedder, FakeLLM, FakeVectorStore

from recallbase_rag.service import NO_RESULTS_TEXT, RagService

BEES_TEXT = "Bees make honey from flower nectar and store the honey in the hive."
CARS_TEXT = "Electric cars use batteries and motors instead of petrol engines."


def make_service(
    store: FakeVectorStore,
    llm: FakeLLM,
) -> RagService:
    return RagService(
        embedder=FakeEmbedder(),
        store=store,
        llm=llm,
        chunk_size=500,
        chunk_overlap=50,
        top_k=2,
    )


def write_file(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_store_document_saves_chunks(tmp_path: Path) -> None:
    store = FakeVectorStore()
    service = make_service(store, FakeLLM())
    path = write_file(tmp_path, "bees.txt", BEES_TEXT)

    count = service.store_document(path)

    assert count == 1
    assert len(store.items) == 1
    assert store.items[0][0].file_name == "bees.txt"


def test_ask_question_returns_answer_and_right_source(tmp_path: Path) -> None:
    store = FakeVectorStore()
    llm = FakeLLM(answer="Bees make honey.")
    service = make_service(store, llm)
    service.store_document(write_file(tmp_path, "bees.txt", BEES_TEXT))
    service.store_document(write_file(tmp_path, "cars.txt", CARS_TEXT))

    answer = service.ask_question("How do bees make honey?")

    assert answer.text == "Bees make honey."
    assert len(answer.sources) == 2
    assert answer.sources[0].chunk.file_name == "bees.txt"
    assert answer.sources[0].score >= answer.sources[1].score


def test_prompt_contains_question_and_source(tmp_path: Path) -> None:
    store = FakeVectorStore()
    llm = FakeLLM()
    service = make_service(store, llm)
    service.store_document(write_file(tmp_path, "bees.txt", BEES_TEXT))

    service.ask_question("How do bees make honey?")

    assert len(llm.prompts) == 1
    assert "How do bees make honey?" in llm.prompts[0].user
    assert "bees.txt" in llm.prompts[0].user


def test_storing_same_file_twice_does_not_duplicate(tmp_path: Path) -> None:
    store = FakeVectorStore()
    service = make_service(store, FakeLLM())
    path = write_file(tmp_path, "bees.txt", BEES_TEXT)

    service.store_document(path)
    service.store_document(path)

    assert len(store.items) == 1


def test_empty_store_skips_the_llm() -> None:
    llm = FakeLLM()
    service = make_service(FakeVectorStore(), llm)

    answer = service.ask_question("Anything there?")

    assert answer.text == NO_RESULTS_TEXT
    assert answer.sources == []
    assert llm.prompts == []


def test_empty_question_is_rejected() -> None:
    service = make_service(FakeVectorStore(), FakeLLM())

    with pytest.raises(ValueError):
        service.ask_question("   ")