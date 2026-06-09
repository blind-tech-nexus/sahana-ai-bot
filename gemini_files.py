import json
import base64
import logging
from typing import Optional

import httpx

from api import extract_ai_text
from api_keys import fetch_api_keys, KeyRotator
from config import USER_MODEL

logger = logging.getLogger("mero.gemini_files")

TRANSCRIBE_PROMPT = (
    "Transcribe the uploaded audio exactly in its original language. "
    "Do not summarize. Do not translate. Return only the transcript text."
)


async def transcribe_audio_inline(
    audio_bytes: bytes,
    mime_type: str,
    model: str = USER_MODEL,
    preferred_key: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Transcribe audio using inline base64 data directly in generateContent.

    This avoids the file upload API entirely — just base64-encode the audio
    and send it as inline_data in the request body. Works for files up to 20MB.
    Uses KeyRotator for proper sequential key iteration.
    """
    if not await fetch_api_keys():
        return None, "No API keys available"

    # Base64-encode the audio bytes
    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_audio,
                        }
                    },
                    {"text": TRANSCRIBE_PROMPT},
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 65536,
            "temperature": 0.0,
        },
    }
    body_json = json.dumps(body)

    rotator = KeyRotator(preferred_key=preferred_key)
    if not rotator.has_keys():
        return None, "No API keys available"

    async with httpx.AsyncClient(timeout=180.0) as client:
        while True:
            key = rotator.get_next_key()
            if key is None:
                break

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={key}"
            )
            try:
                resp = await client.post(
                    url,
                    content=body_json,
                    headers={"Content-Type": "application/json"},
                )
            except Exception as exc:
                rotator.mark_failed(key, f"network_error:{exc.__class__.__name__}")
                continue

            if resp.status_code == 200:
                rotator.mark_success(key)
                text, _ = extract_ai_text(resp.text)
                clean = (text or "").strip()
                if not clean or clean in (
                    "No response received from AI.",
                    "Failed to parse AI response.",
                ):
                    return None, "Empty transcription result"
                return clean, None

            rotator.mark_failed(key, f"status_{resp.status_code}")
            continue

    return None, f"Transcription failed: {rotator.get_failure_summary()}"
