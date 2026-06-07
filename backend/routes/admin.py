"""
admin.py — Admin management endpoints.
"""

import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel

from core.config import get_settings
from ingest.chunker import chunk_parents
from ingest.embedder import embed_chunks, ensure_index, upsert_to_pinecone
from ingest.pdf_loader import build_all_parent_chunks, build_parent_chunks, load_all_pdfs, load_pdf

router = APIRouter(prefix="/admin", tags=["admin"])
settings = get_settings()

openai_client = OpenAI(api_key=settings.openai_api_key)
pc = Pinecone(api_key=settings.pinecone_api_key)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    chunks_indexed: int
    sources: List[str]
    message: str


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


# ---------------------------------------------------------------------------
# Shared ingestion helper
# ---------------------------------------------------------------------------

def _run_ingestion(pages: list) -> IngestResponse:
    if not pages:
        return IngestResponse(
            chunks_indexed=0,
            sources=[],
            message="No pages found to ingest.",
        )

    # Group pages by source then build parent chunks
    sources_map: dict[str, list] = {}
    for page in pages:
        sources_map.setdefault(page["source"], []).append(page)

    all_parents = []
    for source_pages in sources_map.values():
        all_parents.extend(build_parent_chunks(source_pages))

    # Split parents into children
    chunks = chunk_parents(all_parents)

    # Ensure Pinecone index exists
    ensure_index(pc=pc, index_name=settings.pinecone_index)

    # Embed children in batches
    embedded = embed_chunks(
        chunks=chunks,
        client=openai_client,
        model=settings.openai_embedding_model,
        batch_size=100,
    )

    # Upsert to Pinecone kb namespace
    total = upsert_to_pinecone(
        embedded_chunks=embedded,
        pc=pc,
        index_name=settings.pinecone_index,
        namespace="kb",
    )

    sources = list(sources_map.keys())

    return IngestResponse(
        chunks_indexed=total,
        sources=sources,
        message=f"Successfully indexed {total} chunks from {len(sources)} document(s).",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingest() -> IngestResponse:
    pdf_dir = Path(settings.pdf_dir)

    if not pdf_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF directory not found: {pdf_dir}.",
        )

    pages = load_all_pdfs(pdf_dir)

    if not pages:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF files found in {pdf_dir}.",
        )

    return _run_ingestion(pages)


@router.post("/ingest/upload", response_model=IngestResponse)
async def upload_and_ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    content_type = file.content_type or ""
    if content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Invalid content type: '{content_type}'.")

    pdf_dir = Path(settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    save_path = pdf_dir / file.filename

    try:
        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        pages = load_pdf(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {str(e)}")

    return _run_ingestion(pages)


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents() -> List[DocumentInfo]:
    """List all documents in Pinecone with chunk counts."""
    try:
        index = pc.Index(settings.pinecone_index)
        stats = index.describe_index_stats()
        # Pinecone doesn't support per-source counts directly on free tier
        # Return namespace stats instead
        namespaces = stats.get("namespaces", {})
        kb_stats = namespaces.get("kb", {})
        total_vectors = kb_stats.get("vector_count", 0)
        return [DocumentInfo(source="kb", chunk_count=total_vectors)]
    except Exception:
        return []


@router.delete("/documents/{source}")
async def delete_document(source: str) -> dict:
    try:
        index = pc.Index(settings.pinecone_index)
        index.delete(
            filter={"source": {"$eq": source}},
            namespace="kb",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{source}': {str(e)}")

    pdf_path = Path(settings.pdf_dir) / f"{source}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    return {"deleted": True, "source": source}