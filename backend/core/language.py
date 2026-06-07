"""
language.py — Language detection and handling utilities.

Responsibilities:
  - Detect whether a given text is in English, Urdu, or Roman Urdu.
  - Provide helpers to format bilingual system prompts.
  - Centralise language-routing logic consumed by routes and retrieval.
"""

from enum import Enum
from langdetect import detect, LangDetectException


class Language(str, Enum):
    """Supported response languages."""

    ENGLISH    = "en"
    URDU       = "ur"
    ROMAN_URDU = "ro-ur"
    UNKNOWN    = "unknown"


# ---------------------------------------------------------------------------
# Roman Urdu signal words — common vocabulary KUST students actually use
# ---------------------------------------------------------------------------

ROMAN_URDU_SIGNALS = {
    # question words
    "kya", "kab", "kahan", "kyun", "kaise", "kaun",
    # verbs / helpers
    "hai", "hain", "tha", "thi", "ho", "hoga", "hogi",
    "kar", "karo", "karna", "karta", "karti",
    "milega", "milegi", "mile", "chahiye",
    # pronouns
    "mujhe", "meri", "mera", "mere", "aap", "ap",
    "hum", "unka", "uska",
    # university-specific
    "admission", "fees", "deadline", "form", "result",
    "schedule", "timetable", "department", "degree",
    "semester", "exam", "date", "last", "kitni", "kitna",
    # common connectors
    "aur", "ya", "lekin", "bhi", "sirf", "wala", "wali",
    "se", "mein", "pe", "par", "ko", "ka", "ki", "ke",
}

# How many signal word hits before we're confident it's Roman Urdu
ROMAN_URDU_THRESHOLD = 2


def is_urdu_script(text: str) -> bool:
    """
    Determine whether a string contains Urdu / Arabic Unicode characters.
    Urdu script sits in the Unicode Arabic block: U+0600–U+06FF.

    Args:
        text: Input string to inspect.

    Returns:
        True if the text contains characters in the Urdu/Arabic range.
    """
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def detect_language(text: str) -> Language:
    """
    Detect the primary language of a short text string.

    Detection order:
      1. Urdu script check  — fast, zero-cost, handles native Urdu input
      2. Roman Urdu signals — keyword heuristic for transliterated Urdu
      3. langdetect         — statistical model for everything else

    Args:
        text: Input text (query, transcription, etc.).

    Returns:
        A Language enum value.
    """
    if not text or not text.strip():
        return Language.UNKNOWN

    # --- Step 1: Urdu script (fastest check, no ambiguity) ---
    if is_urdu_script(text):
        return Language.URDU

    # --- Step 2: Roman Urdu signal words ---
    tokens = set(text.lower().split())
    hits = tokens & ROMAN_URDU_SIGNALS
    if len(hits) >= ROMAN_URDU_THRESHOLD:
        return Language.ROMAN_URDU

    # --- Step 3: Statistical language detection ---
    try:
        detected = detect(text)
        if detected == "ur":
            return Language.URDU
        if detected == "en":
            return Language.ENGLISH
        # langdetect sometimes tags Roman Urdu as "tl" (Filipino) or "so"
        # due to phonetic similarity — fall back to English in that case
        return Language.ENGLISH
    except LangDetectException:
        return Language.UNKNOWN


def _format_context(context_chunks: list) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.
    Each chunk gets a [Source: X, Page Y] label so the model can cite inline.
    """
    if not context_chunks:
        return "No context available."

    sections = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source", "Unknown")
        page   = chunk.get("page", "?")
        text   = chunk.get("text", "").strip()
        sections.append(f"[{i}] Source: {source}, Page {page}\n{text}")

    return "\n\n---\n\n".join(sections)


def build_system_prompt(language: Language, context_chunks: list) -> str:
    """
    Construct a grounding system prompt in the detected language.

    The prompt instructs the model to answer ONLY from provided context
    and to cite source + page number on every factual claim.

    Args:
        language:       Detected language of the user's query.
        context_chunks: List of RetrievedChunk dicts from the retriever.

    Returns:
        Formatted system prompt string ready to send to the chat model.
    """
    context = _format_context(context_chunks)

    # --- Shared grounding rules (language-independent) ---
    grounding_rules = """
STRICT RULES — follow these without exception:
1. Answer ONLY using the context provided below. Never use outside knowledge.
2. If the answer is not in the context, say so clearly — do not guess.
3. Never fabricate fees, dates, deadlines, or requirements.
4. Always cite your source inline: (Source: <document name>, Page <number>).
5. If multiple chunks support the answer, cite all of them.
"""

    # --- Language-specific instruction ---
    if language == Language.URDU:
        language_instruction = "اردو میں جواب دیں۔ مکمل اردو رسم الخط استعمال کریں۔"

    elif language == Language.ROMAN_URDU:
        language_instruction = (
            "User ne Roman Urdu mein likha hai. "
            "Usi andaaz mein Roman Urdu mein jawab dein. "
            "Urdu script use mat karein jab tak zaruri na ho."
        )

    else:  # ENGLISH or UNKNOWN — default to English
        language_instruction = "Respond in clear, professional English."

    return f"""You are the official AI assistant for Kohat University of Science & Technology (KUST).
You help students and faculty with accurate information about admissions, fees, programs, policies, and university events.

{grounding_rules}

{language_instruction}

--- CONTEXT START ---
{context}
--- CONTEXT END ---"""