"""
transcribe.py — Voice-to-text transcription endpoint using OpenAI Whisper.

Responsibilities:
  - Accept audio file uploads (WAV, MP3, WebM, etc.) from the frontend.
  - Pass audio to OpenAI Whisper (whisper-1) for transcription.
  - Detect and return the language of the transcription.
  - Return the transcript text for downstream chat processing.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile  # Depends removed — auth stripped
from openai import OpenAI
from pydantic import BaseModel

# from auth.dependencies import get_current_user  # auth stripped — uncomment to re-enable
from core.config import get_settings
from core.language import detect_language

router = APIRouter(prefix="/transcribe", tags=["transcribe"])
settings = get_settings()

openai_client = OpenAI(api_key=settings.openai_api_key)

# ---------------------------------------------------------------------------
# Supported MIME types — Whisper accepts all of these
# ---------------------------------------------------------------------------

SUPPORTED_AUDIO_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/mpeg",       # mp3
    "audio/mp4",
    "audio/m4a",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "video/webm",       # browsers often send WebM with video MIME even for audio-only
}

# Max file size: 25MB — Whisper's hard limit
MAX_AUDIO_BYTES = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class TranscriptionResponse(BaseModel):
    transcript: str
    whisper_language: str   # language code detected by Whisper e.g. "en", "ur"
    app_language: str       # language enum from our detector e.g. "en", "ur", "ro-ur"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    # current_user: dict = Depends(get_current_user),  # auth stripped
) -> TranscriptionResponse:
    """
    Transcribe an audio file to text using OpenAI Whisper.

    Accepts any audio format Whisper supports (wav, mp3, m4a, webm, ogg, flac).
    Returns the transcript, Whisper's detected language, and our app's
    language classification (which additionally detects Roman Urdu).

    Args:
        audio: Multipart audio file upload from the frontend mic button.

    Returns:
        TranscriptionResponse with transcript and language metadata.
    """

    # --- Validation: file type ---
    content_type = audio.content_type or ""
    if content_type not in SUPPORTED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported audio format: '{content_type}'. "
                f"Supported formats: wav, mp3, m4a, webm, ogg, flac."
            ),
        )

    # --- Validation: file size ---
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Maximum size is 25MB.",
        )

    if len(audio_bytes) < 100:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty or too short to transcribe.",
        )

    # --- Whisper transcription ---
    # We pass the original filename so Whisper infers the correct codec.
    # No language hint — letting Whisper auto-detect handles both
    # English and Urdu without any extra configuration on our side.
    try:
        response = openai_client.audio.transcriptions.create(
            model=settings.openai_whisper_model,
            file=(audio.filename or "audio.webm", audio_bytes, content_type),
            response_format="verbose_json",  # gives us language + duration metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Whisper transcription failed: {str(e)}",
        )

    transcript = response.text.strip()
    whisper_language = getattr(response, "language", "unknown")

    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="Whisper returned an empty transcript. The audio may be silent or unclear.",
        )

    # --- Our language classifier on top of Whisper ---
    # Whisper detects "ur" for Urdu script but doesn't know Roman Urdu.
    # Our detector catches Roman Urdu that Whisper transcribes as plain text.
    app_language = detect_language(transcript)

    print(
        f"[transcribe] whisper_lang={whisper_language} "
        f"app_lang={app_language} "
        f"transcript='{transcript[:80]}...'"
    )

    return TranscriptionResponse(
        transcript=transcript,
        whisper_language=whisper_language,
        app_language=app_language,
    )