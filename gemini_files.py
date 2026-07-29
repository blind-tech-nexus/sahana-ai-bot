import asyncio
import logging
from typing import Optional

import httpx

from api import call_gemini_raw, normalize_mime_type
from api_keys import KeyRotator, fetch_api_keys, get_next_key_index

logger = logging.getLogger("mero.gemini_files")
TRANSCRIBE_PROMPT = "Transcribe the uploaded audio exactly in its original language. Do not summarize. Do not translate. Return only the transcript text."
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"

async def _upload_with_key(client: httpx.AsyncClient, key: str, file_bytes: bytes, mime_type: str, display_name: str) -> dict:
    start_headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(len(file_bytes)),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    metadata = {"file": {"display_name": display_name}}
    start = await client.post(f"{UPLOAD_URL}?key={key}", headers=start_headers, json=metadata)
    start.raise_for_status()
    upload_url = start.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Gemini Files API did not return an upload URL")
    upload_headers = {
        "Content-Length": str(len(file_bytes)),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }
    final = await client.post(upload_url, headers=upload_headers, content=file_bytes)
    final.raise_for_status()
    return final.json().get("file", final.json())

async def upload_file_bytes(file_bytes: bytes, mime_type: str, display_name: str, max_workers: int = 100) -> tuple[Optional[dict], Optional[str]]:
    """Upload bytes to Gemini Files API using key iteration and return file metadata."""
    if not await fetch_api_keys():
        return None, "No API keys available"
    mime_type = normalize_mime_type(mime_type)
    rotator = KeyRotator(await get_next_key_index())
    tried: list[int] = []
    last_error = None
    limits = httpx.Limits(max_connections=max_workers, max_keepalive_connections=max_workers)
    async with httpx.AsyncClient(timeout=180.0, limits=limits) as client:
        while True:
            idx, key = rotator.get_next_key(tried)
            if key is None:
                break
            try:
                file_obj = await _upload_with_key(client, key, file_bytes, mime_type, display_name)
                if file_obj.get("uri"):
                    return file_obj, None
                last_error = f"Upload response missing file URI: {file_obj}"
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Gemini file upload failed with key index %s: %s", idx, exc)
                await asyncio.sleep(0.2)
    return None, f"All API keys exhausted for file upload. Tried key indexes: {tried}. Last error: {last_error or 'unknown'}"

def file_part(file_obj: dict, fallback_mime: str) -> dict:
    return {"file_data": {"mime_type": normalize_mime_type(file_obj.get("mimeType") or file_obj.get("mime_type") or fallback_mime), "file_uri": file_obj.get("uri") or file_obj.get("fileUri")}}

async def transcribe_audio_inline(audio_bytes: bytes, mime_type: str, chat_id: int) -> tuple[str | None, str | None]:
    mime_type = normalize_mime_type(mime_type)
    uploaded, error = await upload_file_bytes(audio_bytes, mime_type, "voice_audio")
    if not uploaded:
        return None, error or "Gemini Files API upload failed"
    text = await call_gemini_raw(chat_id, [file_part(uploaded, mime_type), {"text": TRANSCRIBE_PROMPT}], "You are an audio transcriber.")
    if not text:
        return None, "Empty transcription result or failed to parse"
    return text.strip(), None
