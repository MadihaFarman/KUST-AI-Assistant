"""
chat.py — Chat completion endpoint.

Responsibilities:
  - Accept a user message (and optional conversation history).
  - Detect language and prepare query for retrieval.
  - Retrieve relevant chunks from Pinecone via the retriever.
  - Stream a GPT-4o-mini answer grounded in the retrieved context.
  - Return inline citations (source filename + page number).
"""

import json
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, HTTPException  # Depends removed — auth stripped
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from pinecone import Pinecone

# from auth.dependencies import get_current_user  # auth stripped — uncomment to re-enable
from core.config import get_settings
from core.language import Language, build_system_prompt, detect_language
from retrieval.query_rewriter import prepare_retrieval_query
from retrieval.retriever import retrieve

router = APIRouter(prefix="/chat", tags=["chat"])
settings = get_settings()


# ---------------------------------------------------------------------------
# Shared clients — instantiated once at module load
# ---------------------------------------------------------------------------

openai_client = OpenAI(api_key=settings.openai_api_key)
pc = Pinecone(api_key=settings.pinecone_api_key)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []
    use_hyde: bool = False          # toggle HyDE per-request for experimentation
    source_filter: Optional[str] = None  # restrict retrieval to one document


class Citation(BaseModel):
    source: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    detected_language: str


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _build_messages(
    system_prompt: str,
    history: List[Message],
    current_query: str,
) -> list:
    """
    Assemble the full message array for the OpenAI chat completion call.

    Structure:
      [system prompt] + [conversation history] + [current user message]

    History is included so the model can resolve pronouns and follow-ups
    like "what about the fees for that department?" correctly.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Include last 6 turns of history maximum — avoids ballooning context costs
    for msg in history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": current_query})
    return messages


def _deduplicate_citations(chunks: list) -> List[Citation]:
    """
    Deduplicate retrieved chunks into unique (source, page) citation pairs.
    Sorted by score descending so the most relevant sources appear first.
    """
    seen = set()
    citations = []
    for chunk in sorted(chunks, key=lambda x: x["score"], reverse=True):
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            citations.append(Citation(
                source=chunk["source"],
                page=chunk["page"],
                score=chunk["score"],
            ))
    return citations


async def _stream_response(messages: list) -> AsyncGenerator[str, None]:
    """
    Stream GPT-4o-mini tokens as Server-Sent Events.
    Each event is a JSON object: {"token": "..."} or {"done": true}.
    """
    stream = openai_client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.1,   # low temp — factual grounded answers, not creative
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield f"data: {json.dumps({'token': delta.content})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    # current_user: dict = Depends(get_current_user),  # auth stripped
) -> StreamingResponse:
    """
    Stream a grounded answer as Server-Sent Events.

    Use this endpoint for the frontend chat UI — tokens arrive word-by-word
    for a responsive typing effect. Citations are sent in the final SSE event.

    Flow:
      1. Detect language of the user's message
      2. Translate + rewrite query for retrieval
      3. Retrieve top-k chunks from Pinecone
      4. Build grounded system prompt
      5. Stream GPT-4o-mini response token by token
      6. Send citations in the closing SSE event
    """
    query = payload.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Step 1: Detect language — used for system prompt and response language
    lang = detect_language(query)
    print(f"[chat] Language detected: {lang} | Query: '{query[:60]}'")

    # Step 2: Prepare retrieval query
    # Translation + rewriting happens here — invisible to the user
    retrieval_query = prepare_retrieval_query(
        query=query,
        client=openai_client,
        model=settings.openai_chat_model,
        use_hyde=payload.use_hyde,
    )

    # Step 3: Retrieve relevant chunks
    chunks = retrieve(
        query=retrieval_query,
        pc=pc,
        openai_client=openai_client,
        index_name=settings.pinecone_index,
        top_k_children=20,
        top_k_parents=5,
        score_threshold=0.35,
        namespace="kb",
    )

    if not chunks:
        # No relevant chunks found — tell the user clearly in their language
        no_info_messages = {
            Language.URDU:       "معذرت، مجھے آپ کے سوال کا جواب اپنی دستاویزات میں نہیں ملا۔",
            Language.ROMAN_URDU: "Sorry, mujhe aap ke sawaal ka jawab apni documents mein nahi mila.",
            Language.ENGLISH:    "I don't have information on that in my knowledge base. Please contact KUST directly.",
        }
        fallback = no_info_messages.get(lang, no_info_messages[Language.ENGLISH])

        async def fallback_stream():
            yield f"data: {json.dumps({'token': fallback})}\n\n"
            yield f"data: {json.dumps({'done': True, 'citations': [], 'language': lang})}\n\n"

        return StreamingResponse(fallback_stream(), media_type="text/event-stream")

    # Step 4: Build grounded system prompt with retrieved context
    system_prompt = build_system_prompt(language=lang, context_chunks=chunks)

    # Step 5: Assemble messages with history
    messages = _build_messages(system_prompt, payload.history, query)

    # Step 6: Build citations before streaming (needed for closing event)
    citations = _deduplicate_citations(chunks)
    citations_payload = [c.model_dump() for c in citations]

    async def event_stream() -> AsyncGenerator[str, None]:
        async for event in _stream_response(messages):
            # Replace the plain {done: true} with an enriched closing event
            if '"done": true' in event:
                closing = {
                    "done": True,
                    "citations": citations_payload,
                    "detected_language": lang,
                }
                yield f"data: {json.dumps(closing)}\n\n"
            else:
                yield event

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    # current_user: dict = Depends(get_current_user),  # auth stripped
) -> ChatResponse:
    """
    Non-streaming chat endpoint — returns the complete answer in one response.

    Use this for testing, the admin panel, or any client that
    doesn't support SSE streaming.
    """
    query = payload.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    lang = detect_language(query)

    retrieval_query = prepare_retrieval_query(
        query=query,
        client=openai_client,
        model=settings.openai_chat_model,
        use_hyde=payload.use_hyde,
    )

    chunks = retrieve(
        query=retrieval_query,
        pc=pc,
        openai_client=openai_client,
        index_name=settings.pinecone_index,
        top_k_children=20,
        top_k_parents=5,
        score_threshold=0.35,
        namespace="kb",
    )

    if not chunks:
        return ChatResponse(
            answer="I don't have information on that in my knowledge base.",
            citations=[],
            detected_language=lang,
        )

    system_prompt = build_system_prompt(language=lang, context_chunks=chunks)
    messages = _build_messages(system_prompt, payload.history, query)

    response = openai_client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=False,
    )

    answer = response.choices[0].message.content.strip()
    citations = _deduplicate_citations(chunks)

    return ChatResponse(
        answer=answer,
        citations=citations,
        detected_language=lang,
    )
