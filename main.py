"""Week 1 v2 demo API: one compact `/ask` endpoint for the intro class.

Run:
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

import vector_store

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")
load_dotenv(THIS_DIR.parent / ".env")

app = FastAPI(title="Week 1 v2 /ask Demo")
_client: OpenAI | None = None

ModelName = Literal["gpt-4o-mini", "gpt-4o", "o3-mini"]
DEFAULT_MODEL: ModelName = "gpt-4o-mini"
DEFAULT_TOP_K = 5
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}

GROUNDING_INSTRUCTIONS = """Answer the question using ONLY the context below. Follow these rules strictly:
1. Do not use any knowledge outside the provided context.
2. Cite the document_id for every claim you make, in the form [document_id].
3. If the context does not contain enough information to answer the question, say so explicitly instead of guessing, and set sources_needed to true."""


class Answer(BaseModel):
    """The model output shape we want every caller to receive."""

    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    model: ModelName | None = None
    force_bad: bool = False
    top_k: int = DEFAULT_TOP_K


class AttemptResult(BaseModel):
    attempt: int
    step: str
    ok: bool
    message: str
    raw_output: str | None = None
    validation_error: str | None = None


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    attempts: list[AttemptResult]
    retrieved_chunk_ids: list[str]


class IngestRequest(BaseModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


class PineconeHealthResponse(BaseModel):
    status: str
    index: str | None = None
    total_vector_count: int | None = None
    detail: str | None = None


class RetrievedChunk(BaseModel):
    id: str
    score: float
    text: str | None = None
    document_id: str | None = None
    source: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    matches: list[RetrievedChunk]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/pinecone")
def debug_pinecone() -> PineconeHealthResponse:
    """Confirms Pinecone is reachable: valid API key, index exists and responds."""
    return PineconeHealthResponse(**vector_store.pinecone_health())


# curl -s "http://127.0.0.1:8000/debug/retrieve?q=What+is+RAG&top_k=5"
#
# Embeds q with text-embedding-3-small and returns the top_k most similar
# chunks from Pinecone with their similarity scores and metadata. Does NOT
# call the LLM - use this to verify retrieval quality before wiring it into
# /ask.
@app.get("/debug/retrieve")
def debug_retrieve(q: str, top_k: int = 5) -> RetrieveResponse:
    if not q.strip():
        raise HTTPException(status_code=400, detail="q must not be empty")

    try:
        matches = vector_store.query_similar(q, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pinecone query failed: {exc}")

    return RetrieveResponse(query=q, matches=[RetrievedChunk(**m) for m in matches])


# curl -s -X POST http://127.0.0.1:8000/ingest \
#   -H "Content-Type: application/json" \
#   -d '{
#         "document_id": "handbook-v1",
#         "text": "Long document text goes here...",
#         "source": "handbook.pdf"
#       }'
#
# Override the default chunk_size/chunk_overlap (env CHUNK_SIZE / CHUNK_OVERLAP,
# default 800/100) per request:
# curl -s -X POST http://127.0.0.1:8000/ingest \
#   -H "Content-Type: application/json" \
#   -d '{"document_id": "handbook-v1", "text": "...", "chunk_size": 500, "chunk_overlap": 50}'
@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunks text with RecursiveCharacterTextSplitter, embeds each chunk with
    text-embedding-3-small, and upserts into Pinecone with document_id/chunk_index/source
    metadata."""
    if not body.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id must not be empty")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    chunks = vector_store.chunk_text(
        body.text, chunk_size=body.chunk_size, chunk_overlap=body.chunk_overlap
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="text produced no chunks after splitting")

    try:
        vector_store.ensure_index()
        chunks_indexed = vector_store.upsert_document_chunks(
            document_id=body.document_id, chunks=chunks, source=body.source
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pinecone ingest failed: {exc}")

    return IngestResponse(
        document_id=body.document_id, chunks_indexed=chunks_indexed, status="indexed"
    )


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_per_1k, output_per_1k = MODEL_PRICES_PER_1K.get(
        model, MODEL_PRICES_PER_1K[DEFAULT_MODEL]
    )
    return (prompt_tokens / 1000 * input_per_1k) + (
        completion_tokens / 1000 * output_per_1k
    )


def usage_counts(completion) -> tuple[int, int, int]:
    usage = completion.usage
    if usage is None:
        return 0, 0, 0
    return usage.total_tokens, usage.prompt_tokens, usage.completion_tokens


def build_grounding_prompt(question: str, chunks: list[dict]) -> str:
    if chunks:
        context = "\n\n".join(
            f"[document_id: {chunk['document_id']}]\n{chunk['text']}" for chunk in chunks
        )
    else:
        context = "(no relevant context was found)"

    return f"{GROUNDING_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}"


def call_structured_model(prompt: str, model: ModelName) -> tuple[Answer, int, int, int]:
    completion = get_client().chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    total_tokens, prompt_tokens, completion_tokens = usage_counts(completion)
    return parsed, total_tokens, prompt_tokens, completion_tokens


def call_malformed_json_once(question: str, model: ModelName) -> tuple[str, int, int, int]:
    """Demo-only path: force one malformed response so students can see retry."""

    completion = get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY JSON using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' instead of a number."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    total_tokens, prompt_tokens, completion_tokens = usage_counts(completion)
    return raw, total_tokens, prompt_tokens, completion_tokens


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    model = body.model or DEFAULT_MODEL
    last_error: str | None = None
    attempts: list[AttemptResult] = []
    total_tokens_used = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    start = time.perf_counter()

    try:
        retrieved = vector_store.query_similar(body.question, top_k=body.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}")

    grounded_prompt = build_grounding_prompt(body.question, retrieved)
    retrieved_chunk_ids = [chunk["id"] for chunk in retrieved]

    for attempt in range(2):
        try:
            if body.force_bad and attempt == 0:
                raw, tokens_used, prompt_tokens, completion_tokens = call_malformed_json_once(
                    body.question, model
                )
                total_tokens_used += tokens_used
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                try:
                    answer = Answer.model_validate_json(raw)
                except ValidationError as exc:
                    last_error = str(exc)
                    attempts.append(
                        AttemptResult(
                            attempt=attempt + 1,
                            step="forced_bad_json",
                            ok=False,
                            message="Validation failed, so the endpoint retries with structured output.",
                            raw_output=raw,
                            validation_error=str(exc),
                        )
                    )
                    continue

                attempts.append(
                    AttemptResult(
                        attempt=attempt + 1,
                        step="forced_bad_json",
                        ok=True,
                        message="Unexpectedly passed validation.",
                        raw_output=raw,
                    )
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_structured_model(
                    grounded_prompt, model
                )
                total_tokens_used += tokens_used
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                attempts.append(
                    AttemptResult(
                        attempt=attempt + 1,
                        step="structured_output",
                        ok=True,
                        message="Structured output matched the Answer schema.",
                    )
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(
                model, total_prompt_tokens, total_completion_tokens
            )
            return AskResponse(
                answer=answer,
                tokens_used=total_tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                attempts=attempts,
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            attempts.append(
                AttemptResult(
                    attempt=attempt + 1,
                    step="structured_output",
                    ok=False,
                    message="Structured output failed validation.",
                    validation_error=str(exc),
                )
            )

    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
