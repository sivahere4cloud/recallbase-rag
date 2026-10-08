import logging
import uuid
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient, models

from recallbase_rag.errors import VectorStoreError
from recallbase_rag.schemas import Chunk, SearchHit

logger = logging.getLogger(__name__)

BATCH_SIZE = 64


class VectorStore(Protocol):
    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]: ...

    def delete_document(self, file_name: str) -> None: ...


def create_qdrant_client(url: str | None, path: Path) -> QdrantClient:
    try:
        if url:
            logger.info("Connecting to Qdrant server at %s", url)
            return QdrantClient(url=url)

        logger.info("Opening local Qdrant storage at %s", path)
        return QdrantClient(path=str(path))
    except Exception as error:
        raise VectorStoreError(f"Could not open Qdrant: {error}") from error


def make_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dimension: int,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension
        self.ensure_collection()

    def ensure_collection(self) -> None:
        try:
            exists = self.client.collection_exists(self.collection_name)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("Created collection %s", self.collection_name)
        except Exception as error:
            raise VectorStoreError(
                f"Could not prepare collection {self.collection_name}: {error}"
            ) from error

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors must have the same length")

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            point = models.PointStruct(
                id=make_point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
            points.append(point)

        try:
            for start in range(0, len(points), BATCH_SIZE):
                batch = points[start : start + BATCH_SIZE]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )
        except Exception as error:
            raise VectorStoreError(f"Could not save chunks: {error}") from error

        logger.info("Saved %s chunks to %s", len(points), self.collection_name)

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as error:
            raise VectorStoreError(f"Search failed: {error}") from error

        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = Chunk(
                chunk_id=payload["chunk_id"],
                file_name=payload["file_name"],
                page_number=payload["page_number"],
                chunk_index=payload["chunk_index"],
                text=payload["text"],
            )
            hits.append(SearchHit(chunk=chunk, score=point.score))

        return hits

    def delete_document(self, file_name: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_name",
                        match=models.MatchValue(value=file_name),
                    )
                ]
            )
        )

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=selector,
            )
        except Exception as error:
            raise VectorStoreError(
                f"Could not delete chunks of {file_name}: {error}"
            ) from error

        logger.info("Deleted chunks of %s", file_name)