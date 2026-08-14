"""Pinecone vector store integration: config, embeddings, ingest, query, health.

All config comes from environment variables - see .env.example. Clients are
created lazily so importing this module never fails just because a key is
missing; you only hit an error when you actually try to use Pinecone.
"""

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

_openai_client: OpenAI | None = None
_pinecone_client: Pinecone | None = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def get_pinecone_client() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return _pinecone_client


def get_index_name() -> str:
    return os.environ["PINECONE_INDEX_NAME"]


def ensure_index() -> None:
    """Create the configured serverless index if it doesn't exist yet."""
    pc = get_pinecone_client()
    name = get_index_name()
    if name not in {idx["name"] for idx in pc.list_indexes()}:
        pc.create_index(
            name=name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.environ.get("PINECONE_CLOUD", "aws"),
                region=os.environ.get("PINECONE_REGION", "us-east-1"),
            ),
        )


def get_index():
    return get_pinecone_client().Index(get_index_name())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed one or more strings with the same model used at ingest and query time."""
    response = get_openai_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_chunk_config() -> tuple[int, int]:
    """Read chunk_size/chunk_overlap from the environment, deferred to call time
    (not module import time) so a value set in .env is picked up correctly."""
    chunk_size = int(os.environ.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    return chunk_size, chunk_overlap


def chunk_text(
    text: str, chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[str]:
    default_size, default_overlap = get_chunk_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or default_size,
        chunk_overlap=chunk_overlap or default_overlap,
    )
    return splitter.split_text(text)


def upsert_document_chunks(
    document_id: str,
    chunks: list[str],
    source: str | None = None,
    namespace: str = "default",
) -> int:
    """Embed each chunk and upsert it, keyed by document_id + chunk_index so
    re-ingesting the same document_id overwrites its previous chunks in place."""
    vectors = embed_texts(chunks)
    payload = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        metadata = {"document_id": document_id, "chunk_index": i, "text": chunk}
        if source:
            metadata["source"] = source
        payload.append((f"{document_id}-{i}", vector, metadata))
    get_index().upsert(vectors=payload, namespace=namespace)
    return len(payload)


def query_similar(question: str, top_k: int = 3, namespace: str = "default") -> list[dict]:
    vector = embed_texts([question])[0]
    result = get_index().query(
        vector=vector, top_k=top_k, namespace=namespace, include_metadata=True
    )
    return [
        {
            "id": match.id,
            "score": match.score,
            "text": (match.metadata or {}).get("text"),
            "document_id": (match.metadata or {}).get("document_id"),
            "source": (match.metadata or {}).get("source"),
        }
        for match in result.matches
    ]


def pinecone_health() -> dict:
    """Reachability check: confirms auth works and the configured index responds.

    Never raises - callers (e.g. a /debug endpoint) get a status field back
    instead of a 500, which is more useful when you're diagnosing config.
    """
    try:
        stats = get_index().describe_index_stats()
        return {
            "status": "ok",
            "index": get_index_name(),
            "total_vector_count": stats.total_vector_count,
        }
    except Exception as exc:
        return {"status": "error", "index": None, "total_vector_count": None, "detail": str(exc)}
