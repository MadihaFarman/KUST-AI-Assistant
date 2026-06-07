"""
embedder.py — OpenAI embedding generation and Pinecone upsert.

Responsibilities:
  - Call OpenAI text-embedding-3-small to embed chunks in batches.
  - Upsert resulting vectors into a Pinecone index.
  - Handle rate-limit retries and batch size limits gracefully.
"""

import time
import hashlib
from typing import List

from openai import OpenAI, RateLimitError, APIError
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Chunk = dict          # matches chunker.Chunk shape
EmbeddedChunk = dict  # Chunk + {"embedding": List[float], "id": str}


def _chunk_to_id(chunk: Chunk) -> str:
    """
    Generate a deterministic ID from chunk content + metadata.
    Same chunk re-ingested always gets the same ID — safe to re-run.
    """
    if chunk.get("child_id"):
        fingerprint = chunk["child_id"]
    else:
        page = chunk.get("page", chunk.get("page_start", 0))
        index = chunk.get("chunk_index", chunk.get("child_index", 0))
        fingerprint = f"{chunk['source']}::p{page}::i{index}::{chunk['text'][:64]}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


def _embed_batch_with_retry(
    texts: List[str],
    client: OpenAI,
    model: str,
    max_retries: int = 3,
) -> List[List[float]]:
    """
    Embed a single batch of texts with exponential backoff on rate limits.
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except RateLimitError:
            wait = 2 ** attempt
            print(f"[embedder] Rate limited. Retrying in {wait}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait)
        except APIError as e:
            print(f"[embedder] API error: {e}. Retrying in 2s...")
            time.sleep(2)

    raise RuntimeError(f"[embedder] Failed to embed batch after {max_retries} retries.")


def embed_chunks(
    chunks: List[Chunk],
    client: OpenAI,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> List[EmbeddedChunk]:
    """
    Generate embeddings for a list of chunks using the OpenAI API.

    Args:
        chunks:     List of Chunk dicts containing at least a "text" field.
        client:     Authenticated OpenAI client instance.
        model:      Embedding model identifier.
        batch_size: Number of chunks to embed per API call (max 2048 for OpenAI).

    Returns:
        List of EmbeddedChunk dicts with an added "embedding" float list
        and a deterministic "id" string.
    """
    if not chunks:
        print("[embedder] No chunks to embed.")
        return []

    embedded = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[start : start + batch_size]
        texts = [chunk["text"] for chunk in batch]

        print(f"[embedder] Embedding batch {batch_num}/{total_batches} ({len(texts)} chunks)...")
        embeddings = _embed_batch_with_retry(texts, client, model)

        for chunk, embedding in zip(batch, embeddings):
            embedded.append({
                **chunk,
                "id": _chunk_to_id(chunk),
                "embedding": embedding,
            })

        # Small polite delay between batches to avoid aggressive rate limiting
        if batch_num < total_batches:
            time.sleep(0.3)

    print(f"[embedder] Done. {len(embedded)} chunks embedded.")
    return embedded


def _chunk_metadata(chunk: EmbeddedChunk) -> dict:
    """Build Pinecone-safe metadata, omitting optional None values."""
    metadata = {
        "text":             chunk["text"],
        "source":           chunk["source"],
        "page":             chunk.get("page", chunk.get("page_start", 0)),
        "page_start":       chunk.get("page_start", chunk.get("page", 0)),
        "page_end":         chunk.get("page_end", chunk.get("page", 0)),
        "total_pages":      chunk.get("total_pages", 0),
        "chunk_index":      chunk.get("chunk_index", chunk.get("child_index", 0)),
        "child_index":      chunk.get("child_index", chunk.get("chunk_index", 0)),
        "parent_id":        chunk.get("parent_id"),
        "child_id":         chunk.get("child_id"),
        "doc_type":         chunk.get("doc_type"),
        "title":            chunk.get("title"),
        "section":          chunk.get("section"),
        "section_title":    chunk.get("section_title"),
        "is_amendment":     chunk.get("is_amendment", False),
        "amends_source":    chunk.get("amends_source"),
        "amends_section":   chunk.get("amends_section"),
        "amendment_action": chunk.get("amendment_action"),
        "token_count":      chunk["token_count"],
    }
    return {key: value for key, value in metadata.items() if value is not None}


def ensure_index(
    pc: Pinecone,
    index_name: str,
    vector_size: int = 1536,
) -> None:
    """
    Create the Pinecone index if it does not already exist.

    Uses the free serverless tier on AWS us-east-1.

    Args:
        pc:          Authenticated Pinecone client instance.
        index_name:  Name of the index to create.
        vector_size: Dimensionality of vectors (1536 for text-embedding-3-small).
    """
    existing = [i.name for i in pc.list_indexes()]

    if index_name in existing:
        print(f"[embedder] Index '{index_name}' already exists. Skipping creation.")
        return

    pc.create_index(
        name=index_name,
        dimension=vector_size,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",   # free tier region
        ),
    )

    # Wait for index to be ready — Pinecone takes a few seconds
    print(f"[embedder] Creating index '{index_name}'... waiting for ready state.")
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    print(f"[embedder] Index '{index_name}' ready (dim={vector_size}, cosine).")


def upsert_to_pinecone(
    embedded_chunks: List[EmbeddedChunk],
    pc: Pinecone,
    index_name: str,
    batch_size: int = 100,
    namespace: str = "kb",
) -> int:
    """
    Upsert embedded chunks as vectors into a Pinecone index.

    Each vector carries its full metadata as Pinecone metadata fields —
    this is what powers source + page citations in retrieval.

    Args:
        embedded_chunks: Output of embed_chunks().
        pc:              Authenticated Pinecone client instance.
        index_name:      Target Pinecone index name.
        batch_size:      Vectors per upsert call.
        namespace:       Pinecone namespace ("kb" for PDF knowledge base,
                         "live" for scraped news/notifications).

    Returns:
        Number of vectors successfully upserted.
    """
    if not embedded_chunks:
        print("[embedder] No embedded chunks to upsert.")
        return 0

    index = pc.Index(index_name)
    total_upserted = 0
    total_batches = (len(embedded_chunks) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(embedded_chunks), batch_size), start=1):
        batch = embedded_chunks[start : start + batch_size]

        # Pinecone upsert format: list of (id, vector, metadata) tuples
        vectors = [
            (
                chunk["id"],
                chunk["embedding"],
                _chunk_metadata(chunk),
            )
            for chunk in batch
        ]

        index.upsert(vectors=vectors, namespace=namespace)
        total_upserted += len(vectors)
        print(f"[embedder] Upserted batch {batch_num}/{total_batches} ({len(vectors)} vectors)")

    print(f"[embedder] Total upserted: {total_upserted} vectors into '{index_name}' (ns={namespace})")
    return total_upserted
