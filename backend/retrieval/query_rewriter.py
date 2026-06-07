"""
query_rewriter.py — LLM-powered query expansion and rewriting.

Responsibilities:
  - Rewrite ambiguous or short queries into retrieval-friendly forms.
  - Translate Urdu / Roman Urdu queries to English before embedding.
  - Implement HyDE (Hypothetical Document Embeddings) for advanced retrieval.
"""

from openai import OpenAI
from core.language import Language, detect_language


def rewrite_query(
    query: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Rewrite a user query to improve retrieval recall.

    Expands abbreviations, resolves pronouns, and adds context keywords
    without changing the user's intent.

    Args:
        query:  Original user question.
        client: Authenticated OpenAI client instance.
        model:  Chat completion model to use for rewriting.

    Returns:
        The rewritten query string.
    """
    # Very short queries (1-2 words) benefit most from rewriting
    # Longer queries are usually specific enough already
    if len(query.split()) > 12:
        return query  # skip rewriting — avoid over-engineering a good query

    response = client.chat.completions.create(
        model=model,
        temperature=0,  # deterministic — we want consistent rewrites
        max_tokens=128,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer for a university knowledge base. "
                    "Rewrite the user's question into a clear, specific, retrieval-friendly query. "
                    "Rules:\n"
                    "- Expand abbreviations (e.g. 'BS CS' → 'Bachelor of Science Computer Science')\n"
                    "- Add relevant context keywords (e.g. 'fee?' → 'fee structure tuition charges')\n"
                    "- Keep the same language as the input (English stays English, Urdu stays Urdu)\n"
                    "- Never change the intent of the question\n"
                    "- Output ONLY the rewritten query. No explanation, no punctuation changes."
                ),
            },
            {
                "role": "user",
                "content": f"Rewrite this query: {query}",
            },
        ],
    )

    rewritten = response.choices[0].message.content.strip()
    print(f"[query_rewriter] Rewrite: '{query}' → '{rewritten}'")
    return rewritten


def translate_to_english(
    text: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Translate Urdu or Roman Urdu text to English for embedding.

    Why: Our PDF knowledge base is in English. Embedding an Urdu query
    against English vectors produces weak similarity scores. Translating
    first aligns the query vector space with the document vector space.

    Args:
        text:   Input text in Urdu or Roman Urdu.
        client: Authenticated OpenAI client instance.
        model:  Chat completion model to use for translation.

    Returns:
        English translation of the input text.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=128,
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate the following Urdu or Roman Urdu text to English. "
                    "Output ONLY the English translation. "
                    "Preserve technical terms like degree names, department names, and proper nouns."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )

    translated = response.choices[0].message.content.strip()
    print(f"[query_rewriter] Translated: '{text[:60]}' → '{translated}'")
    return translated


def hyde_rewrite(
    query: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> str:
    """
    HyDE — Hypothetical Document Embeddings.

    Instead of embedding the raw query, generate a FAKE answer that a
    university document might contain, then embed that instead.

    Why this works:
        A query like "admission last date?" has a thin embedding — it's
        a question, not a document. A hypothetical answer like "The last
        date for undergraduate admissions is March 15th. Students must
        submit..." looks like an actual document chunk and retrieves far
        more relevant results.

    Args:
        query:  Original user question (any language).
        client: Authenticated OpenAI client instance.
        model:  Chat completion model to use.

    Returns:
        A hypothetical answer string to embed in place of the raw query.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,  # slight creativity — we want a plausible document chunk
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are simulating a passage from an official Kohat University (KUST) document. "
                    "Given a student's question, write a short paragraph (3-5 sentences) that a real "
                    "university document might contain as the answer. "
                    "Write in formal English regardless of the query language. "
                    "Do NOT say 'I don't know' — always write a plausible university document excerpt. "
                    "Output ONLY the hypothetical passage. No preamble."
                ),
            },
            {
                "role": "user",
                "content": f"Student question: {query}",
            },
        ],
    )

    hypothesis = response.choices[0].message.content.strip()
    print(f"[query_rewriter] HyDE hypothesis generated ({len(hypothesis.split())} words)")
    return hypothesis


def prepare_retrieval_query(
    query: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    use_hyde: bool = False,
) -> str:
    """
    Full query preparation pipeline before embedding and retrieval.

    Pipeline:
      1. Detect language
      2. Translate to English if Urdu / Roman Urdu
      3. Rewrite for better retrieval recall
      4. Optionally apply HyDE

    Args:
        query:     Raw user query (any language).
        client:    Authenticated OpenAI client instance.
        model:     Chat model for rewriting/translation.
        use_hyde:  If True, apply HyDE instead of standard rewrite.

    Returns:
        Final string to embed for vector search.
    """
    lang = detect_language(query)

    # Step 1: Translate non-English queries to English for embedding
    # The response will still be in the original language — this only
    # affects what we embed for retrieval, not what we show the user
    if lang in (Language.URDU, Language.ROMAN_URDU):
        query_for_retrieval = translate_to_english(query, client, model)
    else:
        query_for_retrieval = query

    # Step 2: Rewrite or HyDE
    if use_hyde:
        return hyde_rewrite(query_for_retrieval, client, model)
    else:
        return rewrite_query(query_for_retrieval, client, model)