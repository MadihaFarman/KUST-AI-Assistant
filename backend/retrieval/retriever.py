"""
retriever.py — Semantic retrieval from Pinecone.
"""

from typing import List

from openai import OpenAI
from pinecone import Pinecone

RetrievedChunk = dict


def embed_query(
    query: str,
    client: OpenAI,
    model: str = "text-embedding-3-small",
) -> List[float]:
    """Embed a user query for Pinecone similarity search."""
    response = client.embeddings.create(
        input=[query.strip()],
        model=model,
    )
    return response.data[0].embedding


def retrieve(
    query: str,
    pc: Pinecone,
    openai_client: OpenAI,
    index_name: str,
    top_k_children: int = 30,   # was 20
    top_k_parents: int = 8,     # was 5
    score_threshold: float = 0.30,  # was 0.35 — slightly lower to catch edge sections
    namespace: str = "kb",
) -> List[RetrievedChunk]:
    """
    Retrieve relevant child chunks from Pinecone and deduplicate by parent_id.

    Child chunks are embedded and searched. Results are grouped by parent_id so
    repeated matches from the same section do not dominate the final context.
    """
    query_vector = embed_query(query, openai_client)
    index = pc.Index(index_name)

    response = index.query(
        vector=query_vector,
        top_k=top_k_children,
        namespace=namespace,
        include_metadata=True,
    )

    best_by_parent: dict[str, RetrievedChunk] = {}

    for match in response.matches:
        if match.score < score_threshold:
            continue

        metadata = match.metadata or {}
        parent_id = metadata.get("parent_id") or match.id
        page = metadata.get("page") or metadata.get("page_start", 0)

        chunk = {
            "text": metadata.get("text", ""),
            "source": metadata.get("source", "unknown"),
            "page": page,
            "page_start": metadata.get("page_start", page),
            "page_end": metadata.get("page_end", page),
            "total_pages": metadata.get("total_pages", 0),
            "parent_id": parent_id,
            "child_id": metadata.get("child_id", match.id),
            "doc_type": metadata.get("doc_type", "general"),
            "title": metadata.get("title", ""),
            "section": metadata.get("section", ""),
            "section_title": metadata.get("section_title", ""),
            "is_amendment": metadata.get("is_amendment", False),
            "amends_source": metadata.get("amends_source"),
            "amends_section": metadata.get("amends_section"),
            "amendment_action": metadata.get("amendment_action"),
            "score": round(match.score, 4),
            "namespace": namespace,
        }

        existing = best_by_parent.get(parent_id)
        if existing is None or chunk["score"] > existing["score"]:
            best_by_parent[parent_id] = chunk

    results = sorted(
        best_by_parent.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k_parents]

    print(
        f"[retriever] '{query[:60]}' → {len(results)} parent(s) "
        f"(top_score={results[0]['score'] if results else 0})"
    )

    return results
