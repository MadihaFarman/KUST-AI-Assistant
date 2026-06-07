"""
chunker.py — Parent-child chunking for RAG ingestion.
"""

import re
from typing import List

import tiktoken

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ParentChunk = dict
ChildChunk  = dict

# ---------------------------------------------------------------------------
# Token settings
# ---------------------------------------------------------------------------

CHILD_TARGET_TOKENS = 350
CHILD_MAX_TOKENS    = 450
CHILD_OVERLAP       = 40

# ---------------------------------------------------------------------------
# Split patterns
# ---------------------------------------------------------------------------

_PARAGRAPH_SPLIT = re.compile(r'\n{2,}')
_CLAUSE_SPLIT    = re.compile(r'(?=\n\(?[a-z]\)\s|\n\(?(?:i{1,3}|iv|vi{0,3}|ix|x)\)\s)', re.IGNORECASE)
_SENTENCE_SPLIT  = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def _merge_short_segments(segments: List[str], encoder: tiktoken.Encoding) -> List[str]:
    merged = []
    buffer = ""
    for seg in segments:
        candidate = (buffer + "\n\n" + seg).strip() if buffer else seg
        if _token_count(candidate, encoder) <= CHILD_MAX_TOKENS:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = seg
    if buffer:
        merged.append(buffer)
    return merged


def _token_slice(text: str, encoder: tiktoken.Encoding) -> List[str]:
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHILD_MAX_TOKENS, len(tokens))
        chunks.append(encoder.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHILD_MAX_TOKENS - CHILD_OVERLAP
    return chunks


def _make_child(text: str, parent: ParentChunk, child_index: int) -> ChildChunk:
    encoder = _get_encoder()
    child_id = f"{parent['parent_id']}::child_{child_index:02d}"
    return {
        "text":             text,
        "child_id":         child_id,
        "parent_id":        parent["parent_id"],
        "source":           parent["source"],
        "doc_type":         parent["doc_type"],
        "title":            parent["title"],
        "section":          parent["section"],
        "section_title":    parent["section_title"],
        "page_start":       parent["page_start"],
        "page_end":         parent["page_end"],
        "is_amendment":     parent.get("is_amendment", False),
        "amends_source":    parent.get("amends_source"),
        "amends_section":   parent.get("amends_section"),
        "amendment_action": parent.get("amendment_action"),
        "child_index":      child_index,
        "token_count":      _token_count(text, encoder),
    }


def split_parent_into_children(parent: ParentChunk, encoder: tiktoken.Encoding) -> List[ChildChunk]:
    text = parent["text"]

    # Case 1: fits in one child
    if _token_count(text, encoder) <= CHILD_MAX_TOKENS:
        return [_make_child(text, parent, 0)]

    # Case 2: paragraph splits
    segments = _merge_short_segments(
        [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()], encoder
    )
    if len(segments) > 1 and all(_token_count(s, encoder) <= CHILD_MAX_TOKENS for s in segments):
        return [_make_child(s, parent, i) for i, s in enumerate(segments)]

    # Case 3: clause splits
    segments = _merge_short_segments(
        [p.strip() for p in _CLAUSE_SPLIT.split(text) if p.strip()], encoder
    )
    if len(segments) > 1 and all(_token_count(s, encoder) <= CHILD_MAX_TOKENS for s in segments):
        return [_make_child(s, parent, i) for i, s in enumerate(segments)]

    # Case 4: sentence splits
    segments = _merge_short_segments(
        [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()], encoder
    )
    if len(segments) > 1 and all(_token_count(s, encoder) <= CHILD_MAX_TOKENS for s in segments):
        return [_make_child(s, parent, i) for i, s in enumerate(segments)]

    # Case 5: token slice fallback
    return [_make_child(s, parent, i) for i, s in enumerate(_token_slice(text, encoder))]


def chunk_parents(parents: List[ParentChunk]) -> List[ChildChunk]:
    """Main entry point — split all parent chunks into children."""
    if not parents:
        print("[chunker] No parent chunks to split.")
        return []

    encoder = _get_encoder()
    all_children = []

    for parent in parents:
        children = split_parent_into_children(parent, encoder)
        all_children.extend(children)

    avg = sum(c["token_count"] for c in all_children) // len(all_children) if all_children else 0
    print(f"[chunker] {len(parents)} parents → {len(all_children)} children (avg {avg} tokens)")
    return all_children