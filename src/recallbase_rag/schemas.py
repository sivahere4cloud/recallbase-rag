from pydantic import BaseModel, ConfigDict, Field


class Page(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_name: str
    page_number: int = Field(ge=1)
    text: str


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    file_name: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    text: str
    sources: list[SearchHit]