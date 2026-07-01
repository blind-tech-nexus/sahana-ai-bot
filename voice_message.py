from database import save_message, get_all_history, get_user_voice, get_file_data
from message import send_message, download_telegram_file, send_chat_action, send_voice_bytes
from transcriber import transcribe_audio_bytes
from api import handle_gemini
from system import get_system_text
from tts import generate_tts


def _voice_mime_type(voice: dict) -> str:
    mime = (voice.get("mime_type") or "").strip().lower()
    if not mime:
        return "audio/ogg"
    if "/" not in mime:
        return "audio/ogg"
    return mime


async def handle_voice(cid: int, voice: dict, name: str) -> None:
    if voice.get("duration", 0) > 300:
        await send_message(cid, "⚠️ Voice messages up to 5 minutes only.")
        return
    await send_chat_action(cid, "typing")
    await send_message(cid, "🎙️ Processing voice...")
    voice_data = await download_telegram_file(voice["file_id"])
    if not voice_data:
        await send_message(cid, "❌ Failed to download voice message.")
        return
    mime_type = _voice_mime_type(voice)
    transcription, _ = await transcribe_audio_bytes(voice_data, mime_type, "voice.ogg", chat_id=cid)
    transcription_text = (transcription or "").strip()
    if not transcription_text or transcription_text in ("No response received from AI.", "Failed to parse AI response."):
        await send_message(cid, "❌ Failed to transcribe voice message.")
        return
    save_message(cid, "user", f"[Voice] {transcription_text}")
    await send_message(cid, f"📝 Transcribed: {transcription_text}")

    # Send transcribed text directly to Gemini (with function calling support)
    current_parts: list = [{"text": transcription_text}]
    file_data = get_file_data(cid)
    has_file = False
    if file_data and file_data.get("base64"):
        current_parts.append({"inlineData": {"mimeType": file_data["mime_type"], "data": file_data["base64"]}})
        has_file = True
    await handle_gemini(
        cid,
        current_parts,
        get_system_text(name, cid),
        use_tools=not has_file,
        user_name=name,
    )

    # Generate voice response for the AI reply
    history = await get_all_history(cid)
    if not history:
        return
    last = history[-1]
    if last.get("role") != "model":
        return
    reply_text = (last.get("text") or "").strip()
    if not reply_text:
        return
    voice_lang = await get_user_voice(cid)
    voice_audio = await generate_tts(reply_text, voice_lang)
    if voice_audio:
        await send_voice_bytes(cid, voice_audio, "🎧 Voice response", "response.mp3", "audio/mpeg")
