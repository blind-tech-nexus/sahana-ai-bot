import base64
from api import call_gemini_raw, normalize_mime_type, upload_file_with_retry, delete_gemini_file, MAX_INLINE_FILE_BYTES, try_api_call, get_gemini_model, extract_ai_text

TRANSCRIBE_PROMPT = (
    "Transcribe the uploaded audio exactly in its original language. "
    "Do not summarize. Do not translate. Return only the transcript text."
)


async def transcribe_audio_inline(audio_bytes: bytes, mime_type: str, chat_id: int) -> tuple[str | None, str | None]:
    mime_type = normalize_mime_type(mime_type)
    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

    parts = [
        {"inlineData": {"mimeType": mime_type, "data": encoded_audio}},
        {"text": TRANSCRIBE_PROMPT}
    ]

    text = await call_gemini_raw(chat_id, parts, "You are an audio transcriber.")
    if not text or text in ("No response received from AI.", "Failed to parse AI response."):
        return None, "Empty transcription result or failed to parse"
    return text.strip(), None


async def transcribe_audio_file_api(audio_bytes: bytes, mime_type: str, display_name: str = "audio", chat_id: int = 0) -> tuple[str | None, str | None]:
    """Transcribe audio using the Gemini Files API for larger files.

    Uses Files API upload for files > 2MB, falling back to inline for smaller.
    Files are uploaded, used for inference, then deleted to manage storage quota.
    """
    mime_type = normalize_mime_type(mime_type)

    if len(audio_bytes) > MAX_INLINE_FILE_BYTES:
        file_uri = await upload_file_with_retry(audio_bytes, mime_type, display_name)
        if not file_uri:
            return None, "Failed to upload audio file to Gemini Files API"

        model = await get_gemini_model(chat_id)
        body = {
            "systemInstruction": {"parts": [{"text": "You are an audio transcriber."}]},
            "contents": [{"role": "user", "parts": [{"fileUri": file_uri, "mimeType": mime_type}, {"text": TRANSCRIBE_PROMPT}]}],
            "generationConfig": {"maxOutputTokens": 64000, "temperature": 0.4},
        }
        data, err = await try_api_call(model, body)

        await delete_gemini_file(file_uri)

        if not data:
            return None, err or "Failed to transcribe audio"
        text, _ = extract_ai_text(data)
        if not text or text in ("No response received from AI.", "Failed to parse AI response."):
            return None, "Empty transcription result or failed to parse"
        return text.strip(), None

    return await transcribe_audio_inline(audio_bytes, mime_type, chat_id)
