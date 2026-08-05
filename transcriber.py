from typing import Optional

from gemini_files import transcribe_audio_inline, transcribe_audio_file_api
from api import MAX_INLINE_FILE_BYTES
from message import download_telegram_file, send_document_bytes, send_message

MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20MB limit for Telegram file download


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    mime_type: str,
    display_name: str = "audio",
    chat_id: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Transcribe audio bytes using Gemini.

    Uses the Files API for files larger than MAX_INLINE_FILE_BYTES (2MB),
    which is the recommended approach per Gemini API best practices.
    """
    cid = chat_id if chat_id is not None else 0

    if len(audio_bytes) > MAX_INLINE_FILE_BYTES:
        return await transcribe_audio_file_api(audio_bytes, mime_type, display_name, chat_id=cid)

    return await transcribe_audio_inline(
        audio_bytes,
        mime_type,
        chat_id=cid,
    )


async def transcribe_from_telegram_message(cid: int, message: dict) -> bool:
    voice = message.get("voice")
    audio = message.get("audio")
    document = message.get("document")

    if voice:
        file_id = voice.get("file_id")
        file_size = int(voice.get("file_size", 0) or 0)
        mime_type = voice.get("mime_type") or "audio/ogg"
        display_name = "voice.ogg"
    elif audio:
        file_id = audio.get("file_id")
        file_size = int(audio.get("file_size", 0) or 0)
        mime_type = audio.get("mime_type") or "audio/mpeg"
        display_name = audio.get("file_name") or "audio"
    elif document:
        file_id = document.get("file_id")
        file_size = int(document.get("file_size", 0) or 0)
        mime_type = document.get("mime_type") or "application/octet-stream"
        display_name = document.get("file_name") or "audio"
    else:
        await send_message(cid, "❌ Please upload a valid voice or audio file.")
        return False

    if file_size > MAX_AUDIO_BYTES:
        await send_message(cid, "⚠️ Audio must be under 20 MB for inline transcription.")
        return False

    await send_message(cid, "🎙️ Transcribing your audio...")
    audio_bytes = await download_telegram_file(file_id)
    if not audio_bytes:
        await send_message(cid, "❌ Failed to download your audio.")
        return False

    transcription, error = await transcribe_audio_bytes(
        audio_bytes, mime_type, display_name, chat_id=cid
    )
    if error or not transcription:
        await send_message(cid, f"❌ Transcription failed. {error or ''}".strip())
        return False

    if len(transcription) > 4000:
        await send_document_bytes(
            cid,
            transcription.encode("utf-8"),
            "transcription.txt",
            "✅ Transcription is attached as text file.",
            mime_type="text/plain",
        )
    else:
        await send_message(cid, transcription)
    return True
