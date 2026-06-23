import json
import base64
import logging
from typing import Optional

import httpx

from api import extract_ai_text, normalize_mime_type, get_gemini_model
from api_keys import fetch_api_keys, get_next_key_index, KeyRotator, is_retriable_error

logger = logging.getLogger("mero.gemini_files")

TRANSCRIBE_PROMPT = (
    "Transcribe the uploaded audio exactly in its original language. "
    "Do not summarize. Do not translate. Return only the transcript text."
)


async def transcribe_audio_inline(
    audio_bytes: bytes,
    mime_type: str,
    chat_id: int,
) -> tuple[Optional[str], Optional[str]]:
    """Transcribe audio using inline base64 data directly in generateContent.

    This avoids the file upload API entirely — just base64-encode the audio
    and send it as inline_data in the request body. Works for files up to 20MB.
    Uses KeyRotator for proper sequential key iteration.
    """
    if not await fetch_api_keys():
        return None, "No API keys available"

    model = get_gemini_model(chat_id)
    mime_type = normalize_mime_type(mime_type)
    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
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

    start_idx = await get_next_key_index()
    rotator = KeyRotator(start_idx)

    last_error = None
    async with httpx.AsyncClient(
        timeout=180.0,
        limits=httpx.Limits(max_connections=500, max_keepalive_connections=100)
    ) as client:
        while True:
            key = rotator.get_next_key()
            if key is None:
                break

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            try:
                resp = await client.post(
                    url,
                    content=body_json,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    text, _ = extract_ai_text(resp.text)
                    clean = (text or "").strip()
                    if not clean or clean in (
                        "No response received from AI.",
                        "Failed to parse AI response.",
                    ):
                        last_error = "Empty transcription result or failed to parse"
                        continue
                    return clean, None

                logger.warning("Transcription API call failed with status %d: %s", resp.status_code, resp.text)
                last_error = f"Status {resp.status_code}: {resp.text}"
            except Exception as exc:
                logger.warning("Transcription API call exception: %s", exc)
                last_error = str(exc)

    return None, f"Transcription failed: {last_error or 'All keys exhausted'}"
