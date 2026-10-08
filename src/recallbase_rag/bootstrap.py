from functools import lru_cache

from recallbase_rag.config import Settings, get_settings
from recallbase_rag.embedder import SentenceTransformerEmbedder
from recallbase_rag.errors import LLMError
from recallbase_rag.llm import OpenAILLM
from recallbase_rag.logging_setup import setup_logging
from recallbase_rag.service import RagService
from recallbase_rag.vector_store import QdrantVectorStore, create_qdrant_client


def build_service(settings: Settings) -> RagService:
    embedder = SentenceTransformerEmbedder(
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension,
    )

    client = create_qdrant_client(settings.qdrant_url, settings.qdrant_path)
    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        dimension=settings.embedding_dimension,
    )

    if settings.openai_api_key is None:
        raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.")
    llm = OpenAILLM(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
    )

    return RagService(
        embedder=embedder,
        store=store,
        llm=llm,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )


@lru_cache
def get_service() -> RagService:
    settings = get_settings()
    setup_logging(settings.log_level)
    return build_service(settings)