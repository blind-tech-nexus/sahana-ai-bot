from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import ADMINS, TEMPLATE_PROMPTS, SHARE_TEXT, MAX_HISTORY
from database import (
    save_user, user_exists, remove_all_user_data, get_all_users,
    ban_user, unban_user, is_banned, get_banned_users,
    save_message, get_all_history, clear_history,
    set_reply_state, get_reply_state, clear_reply_state,
    set_state, get_state, clear_state,
    save_file_data, get_file_data, clear_file_data,
    get_user_voice, set_user_voice, get_user_system, set_user_system,
    clear_user_system, get_user_temp, set_user_temp,
    get_credit_message, set_credit_message,
    get_memories, save_memory, clear_memories,
    ensure_user, is_admin, check_banned,
    get_user_model, set_user_model,
)
from message import (
    send_message, send_photo, send_voice_bytes,
    download_telegram_file, get_telegram_file_info,
    answer_callback, edit_message, delete_message,
    send_chat_action, send_document_bytes, copy_message,
)
from settings import (
    btn, url_btn, ikb,
    start_keyboard, template_prompts_keyboard,
    user_settings_keyboard, admin_settings_keyboard,
    voice_keyboard, temp_keyboard, model_keyboard, photo_keyboard, file_prompt_keyboard, language_name,
    admin_reply_keyboard, admin_user_reply_keyboard, broadcast_reply_keyboard,
    share_keyboard,
)
from markdown_parse import escape_html
from api import handle_gemini
from system import get_system_text
from image_generation import execute_image
from voice_message import handle_voice
from transcriber import transcribe_from_telegram_message
from group_hooks import is_group_chat, extract_group_prompt
from attachment import (
    handle_photo, handle_document, handle_audio,
    handle_video, handle_animation, handle_sticker,
)

from tools import (
    open_tools_menu,
    run_text_refiner,
    run_text_translator,
    run_pdf_creator,
    run_text_analyzer,
    parse_text_document_bytes,
    resolve_language,
    TOOL_CANCEL,
    MAX_TOOL_TEXT_FILE_BYTES,
)

import urllib.parse
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()
logger = logging.getLogger("mero.main")


def get_user_name(message: dict) -> str:
    user = message.get("from", {})
    name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
    return name or user.get("username", "User")


async def send_banned_message(cid: int) -> None:
    await send_message(
        cid,
        "🚫 Sorry, you're banned by the admin. You can't use the bot anymore.\n\nYou can request the admin to unban you.",
        reply_markup=ikb([[btn("📩 Request Unban", "request_unban")]]),
    )


def get_username(uid: int) -> str:
    return get_all_users().get(str(uid), str(uid))


def _set_broadcast_failed(admin_id: int, user_ids: list[int]) -> None:
    set_state(admin_id, "broadcast_failed:" + ",".join(str(i) for i in user_ids))


def _get_broadcast_failed(admin_id: int) -> list[int]:
    st = get_state(admin_id) or ""
    if not st.startswith("broadcast_failed:"):
        return []
    raw = st.split(":", 1)[1].strip()
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip().lstrip("-").isdigit()]


def _in_tool(st: str | None) -> bool:
    return bool(st and st.startswith("tool:"))


async def send_feedback_to_admins(sender_id: int, sender_name: str, text: str) -> None:
    safe_name = escape_html(sender_name)
    msg = (
        f"📬 <b>New Feedback</b>\n\n"
        f"👤 <b>From:</b> {safe_name}\n"
        f"🆔 <b>ID:</b> <code>{sender_id}</code>\n\n"
        f"💬 {escape_html(text)}"
    )
    for admin_id in ADMINS:
        await send_message(admin_id, msg, parse_mode="HTML", reply_markup=admin_user_reply_keyboard(sender_id, sender_name))


async def relay_voice_reply(sender_id: int, sender_name: str, target_id: int, voice: dict, admin_origin: bool) -> bool:
    voice_data = await download_telegram_file(voice["file_id"])
    if not voice_data:
        await send_message(sender_id, "❌ Failed to download voice message.")
        return False
    if admin_origin:
        caption = f"🎙️ Voice from Admin"
        result = await send_voice_bytes(target_id, voice_data, caption, "admin_voice.ogg", voice.get("mime_type", "audio/ogg"))
        status = "✅ Sent" if result and result.get("ok") else "❌ Failed"
        await send_message(sender_id, f"{status} voice to <code>{target_id}</code>.", parse_mode="HTML")
        return bool(result and result.get("ok"))
    caption = f"🎙️ Voice reply from {escape_html(sender_name)} (<code>{sender_id}</code>)"
    for admin_id in ADMINS:
        await send_voice_bytes(admin_id, voice_data, None, "user_voice.ogg", voice.get("mime_type", "audio/ogg"))
        await send_message(admin_id, caption, parse_mode="HTML", reply_markup=admin_user_reply_keyboard(sender_id, sender_name))
    await send_message(sender_id, "✅ Voice reply sent!")
    return True


async def run_broadcast(admin_id: int, text: str | None = None, voice_data: bytes | None = None, voice_mime: str = "audio/ogg") -> list[int]:
    users = [int(uid) for uid in get_all_users()]
    if voice_data is not None:
        success, fail, failed_ids = _run_parallel_broadcast(users, lambda target: _send_broadcast_voice_sync(target, voice_data, voice_mime))
    else:
        success, fail, failed_ids = _run_parallel_broadcast(users, lambda target: _send_broadcast_text_sync(target, text or ""))
    if failed_ids:
        await send_message(admin_id, f"📢 Broadcast done.\n✅ Sent: {success}\n❌ Failed: {fail}", reply_markup=ikb([[btn("🧹 Clear failed users", f"broadcast_clear_failed:{admin_id}")]]))
    else:
        await send_message(admin_id, f"📢 Broadcast done.\n✅ Sent: {success}\n❌ Failed: {fail}")
    return failed_ids


async def run_broadcast_copy(admin_id: int, source_chat_id: int, source_message_id: int) -> list[int]:
    users = [int(uid) for uid in get_all_users()]
    success, fail, failed_ids = _run_parallel_broadcast(users, lambda target: _copy_broadcast_message_sync(target, source_chat_id, source_message_id))
    if failed_ids:
        await send_message(admin_id, f"📢 Attachment broadcast done.\n✅ Sent: {success}\n❌ Failed: {fail}", reply_markup=ikb([[btn("🧹 Clear failed users", f"broadcast_clear_failed:{admin_id}")]]))
    else:
        await send_message(admin_id, f"📢 Attachment broadcast done.\n✅ Sent: {success}\n❌ Failed: {fail}")
    return failed_ids


def _send_broadcast_text_sync(target: int, text: str) -> bool:
    result = asyncio.run(
        send_message(
            target,
            f"📢 <b>Broadcast:</b>\n\n{escape_html(text)}",
            parse_mode="HTML",
            reply_markup=broadcast_reply_keyboard(),
        )
    )
    return bool(result and result.get("ok"))


def _send_broadcast_voice_sync(target: int, voice_data: bytes, voice_mime: str) -> bool:
    voice_result = asyncio.run(send_voice_bytes(target, voice_data, "📢 Voice broadcast", "broadcast.ogg", voice_mime))
    if not voice_result or not voice_result.get("ok"):
        return False
    message_result = asyncio.run(send_message(target, "📢 <b>Voice Broadcast</b>", parse_mode="HTML", reply_markup=broadcast_reply_keyboard()))
    return bool(message_result and message_result.get("ok"))


def _copy_broadcast_message_sync(target: int, source_chat_id: int, source_message_id: int) -> bool:
    result = asyncio.run(
        copy_message(
            to_chat_id=target,
            from_chat_id=source_chat_id,
            message_id=source_message_id,
            reply_markup=broadcast_reply_keyboard(),
        )
    )
    return bool(result and result.get("ok"))


def _run_parallel_broadcast(users: list[int], sender) -> tuple[int, int, list[int]]:
    success, fail = 0, 0
    failed_ids: list[int] = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(sender, target): target for target in users}
        for future in as_completed(futures):
            target = futures[future]
            try:
                if future.result():
                    success += 1
                else:
                    fail += 1
                    failed_ids.append(target)
            except Exception:
                fail += 1
                failed_ids.append(target)
    return success, fail, failed_ids


@app.get("/")
async def home():
    return {"status": "ok", "message": "Sahana AI is running!"}


@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()

        if "callback_query" in data:
            cb = data["callback_query"]
            cb_id = cb["id"]
            cb_data = cb.get("data", "")
            cb_message = cb.get("message") or {}
            cid = cb_message.get("chat", {}).get("id") or cb.get("from", {}).get("id")
            mid = cb_message.get("message_id")
            if cid is None:
                await answer_callback(cb_id, "Invalid callback context.")
                logger.warning("callback_missing_chat_id data=%s", cb_data)
                return JSONResponse({"ok": True})
            name = get_user_name(cb)

            if check_banned(cid) and cb_data != "request_unban":
                await answer_callback(cb_id, "You are banned.")
                return JSONResponse({"ok": True})

            if cb_data == "noop":
                await answer_callback(cb_id, "Processing...")
                return JSONResponse({"ok": True})

            if cb_data == "share_bot":
                await answer_callback(cb_id)
                await send_message(
                    cid,
                    f"📤 <b>Share Sahana AI with your friends!</b>\n\n{escape_html(SHARE_TEXT)}",
                    parse_mode="HTML",
                    reply_markup=share_keyboard(),
                )
                return JSONResponse({"ok": True})

            if cb_data == "open_settings":
                await answer_callback(cb_id)
                kb = admin_settings_keyboard() if is_admin(cid) else user_settings_keyboard()
                await send_message(cid, "⚙️ <b>Settings</b>", parse_mode="HTML", reply_markup=kb)
                return JSONResponse({"ok": True})

            if cb_data == "open_tools":
                await answer_callback(cb_id)
                set_state(cid, "tool:menu")
                await open_tools_menu(cid)
                return JSONResponse({"ok": True})

            if cb_data == "tools_close":
                await answer_callback(cb_id, "Tools closed")
                st_now = get_state(cid) or ""
                if st_now.startswith("tool:") and st_now != "tool:menu":
                    set_state(cid, "tool:menu")
                    await delete_message(cid, mid)
                    await open_tools_menu(cid)
                else:
                    clear_state(cid)
                    await delete_message(cid, mid)
                    await send_message(cid, "🧰 Tools menu closed.")
                return JSONResponse({"ok": True})

            if cb_data == "tools_cancel":
                await answer_callback(cb_id, "Cancelled")
                clear_state(cid)
                set_state(cid, "tool:menu")
                await open_tools_menu(cid)
                return JSONResponse({"ok": True})

            if cb_data.startswith("tool:"):
                await answer_callback(cb_id)
                tool_name = cb_data.split(":", 1)[1]
                if tool_name == "text_refiner":
                    set_state(cid, "tool:text_refiner")
                    await send_message(cid, "Write or upload your text. You can upload (.txt) file upto 30 kb to refine.", reply_markup=TOOL_CANCEL)
                elif tool_name == "text_translator":
                    set_state(cid, "tool:text_translator:text")
                    await send_message(cid, "Send or upload your text to translate. You can upload .txt file to translate.", reply_markup=TOOL_CANCEL)
                elif tool_name == "pdf_creator":
                    set_state(cid, "tool:pdf_creator")
                    await send_message(cid, "Write a topic or subject to create a pdf with AI.", reply_markup=TOOL_CANCEL)
                elif tool_name == "text_analyzer":
                    set_state(cid, "tool:text_analyzer")
                    await send_message(cid, "Write or upload your .txt to analyze.", reply_markup=TOOL_CANCEL)
                elif tool_name == "audio_transcriber":
                    set_state(cid, "tool:audio_transcriber")
                    await send_message(cid, "Upload your audio. I am ready to transcribe.", reply_markup=TOOL_CANCEL)
                return JSONResponse({"ok": True})

            if cb_data == "request_unban":
                await answer_callback(cb_id, "Request sent!")
                for admin_id in ADMINS:
                    await send_message(
                        admin_id,
                        f"📩 <b>Unban Request</b>\n\n👤 {escape_html(name)}\n🆔 <code>{cid}</code>",
                        parse_mode="HTML",
                        reply_markup=ikb([
                            [btn("✅ Unban", f"do_unban:{cid}"), btn("❌ Reject", f"reject_unban:{cid}")],
                        ]),
                    )
                await send_message(cid, "📩 Your unban request has been sent to the admin.")
                return JSONResponse({"ok": True})

            if cb_data.startswith("do_unban:"):
                if not is_admin(cid):
                    await answer_callback(cb_id, "Unauthorized")
                    return JSONResponse({"ok": True})
                target = int(cb_data.split(":")[1])
                unban_user(target)
                await answer_callback(cb_id, "Unbanned!")
                await edit_message(cid, mid, f"✅ User <code>{target}</code> has been unbanned.", parse_mode="HTML")
                await send_message(target, "🎉 You have been unbanned! You can use the bot again. Send /start")
                return JSONResponse({"ok": True})

            if cb_data.startswith("reject_unban:"):
                if not is_admin(cid):
                    await answer_callback(cb_id, "Unauthorized")
                    return JSONResponse({"ok": True})
                target = int(cb_data.split(":")[1])
                await answer_callback(cb_id, "Rejected")
                await edit_message(cid, mid, f"❌ Unban request from <code>{target}</code> rejected.", parse_mode="HTML")
                await send_message(target, "❌ Your unban request was rejected by the admin.")
                return JSONResponse({"ok": True})

            if cb_data.startswith("tp:"):
                idx = int(cb_data.split(":")[1])
                if 0 <= idx < len(TEMPLATE_PROMPTS):
                    await answer_callback(cb_id)
                    ensure_user(cid, name)
                    await send_chat_action(cid, "typing")
                    prompt_text = TEMPLATE_PROMPTS[idx]
                    save_message(cid, "user", prompt_text)
                    await handle_gemini(
                        cid,
                        [{"text": prompt_text}],
                        get_system_text(name, cid),
                        user_name=name,
                    )
                return JSONResponse({"ok": True})

            if cb_data == "describe_photo":
                await answer_callback(cb_id)
                fd = get_file_data(cid)
                if not fd:
                    await send_message(cid, "❌ No image found. Please send an image first.")
                    return JSONResponse({"ok": True})
                ensure_user(cid, name)
                await send_chat_action(cid, "typing")
                prompt = "Describe this image in detail."
                save_message(cid, "user", f"[Image] {prompt}")
                parts: list = [{"text": prompt}]
                if fd.get("base64"):
                    parts.append({"inlineData": {"mimeType": fd["mime_type"], "data": fd["base64"]}})
                await handle_gemini(
                    cid,
                    parts,
                    get_system_text(name, cid),
                    use_tools=False,
                )
                return JSONResponse({"ok": True})

            if cb_data == "cancel_attachment":
                await answer_callback(cb_id, "Cancelled")
                clear_file_data(cid)
                clear_state(cid)
                await edit_message(cid, mid, "✅ Attachment cancelled.")
                return JSONResponse({"ok": True})

            if cb_data == "export_chat":
                await answer_callback(cb_id)
                history = get_all_history(cid)
                if not history:
                    await send_message(cid, "📜 No conversation history to export.")
                    return JSONResponse({"ok": True})
                export_lines = []
                for msg in history:
                    role = "You" if msg["role"] == "user" else "Sahana AI"
                    export_lines.append(f"[{role}]\n{msg.get('text', '')}\n")
                export_text = "\n---\n".join(export_lines)
                file_bytes = export_text.encode("utf-8")
                await send_document_bytes(cid, file_bytes, "chat_history.txt", "📜 Your chat history export")
                return JSONResponse({"ok": True})

            if cb_data == "clear":
                await answer_callback(cb_id)
                await send_message(cid, "⚠️ Clear chat history?", reply_markup=ikb([
                    [btn("✅ Yes", "clear_yes"), btn("❌ No", "clear_no")],
                ]))
                return JSONResponse({"ok": True})

            if cb_data == "clear_yes":
                await answer_callback(cb_id, "Cleared!")
                clear_history(cid)
                clear_memories(cid)
                await edit_message(cid, mid, "🗑️ Conversation cleared.")
                return JSONResponse({"ok": True})

            if cb_data == "memory_settings":
                await answer_callback(cb_id)
                memories = get_memories(cid)
                text = "🧠 <b>Saved Memories</b>\n\n"
                text += "\n".join(f"{idx + 1}. {escape_html(item)}" for idx, item in enumerate(memories)) if memories else "No memories saved."
                await send_message(
                    cid,
                    text,
                    parse_mode="HTML",
                    reply_markup=ikb([[btn("➕ Add Memory", "memory_add")], [btn("🗑️ Clear Memories", "memory_clear")], [btn("🔙 Back", "back_settings")]]),
                )
                return JSONResponse({"ok": True})

            if cb_data == "memory_add":
                await answer_callback(cb_id)
                set_state(cid, "awaiting_memory_add")
                await send_message(cid, "Send memory text to save forever.", reply_markup=ikb([[btn("❌ Cancel", "cancel_reply")]]))
                return JSONResponse({"ok": True})

            if cb_data == "memory_clear":
                await answer_callback(cb_id, "Cleared")
                clear_memories(cid)
                await send_message(cid, "✅ Memories cleared.")
                return JSONResponse({"ok": True})

            if cb_data == "clear_no":
                await answer_callback(cb_id, "Cancelled")
                await edit_message(cid, mid, "✅ Clear cancelled.")
                return JSONResponse({"ok": True})

            if cb_data == "cls":
                await answer_callback(cb_id, "Attachment cleared!")
                clear_file_data(cid)
                await send_message(cid, "🧹 Stored attachment cleared.")
                return JSONResponse({"ok": True})

            if cb_data == "feedback_prompt":
                await answer_callback(cb_id)
                set_reply_state(cid, -1)
                await send_message(cid, "💬 Send feedback as text, voice, photo, document, audio, video, animation, or sticker.", reply_markup=ikb([[btn("❌ Cancel", "cancel_reply")]]))
                return JSONResponse({"ok": True})

            if cb_data == "cancel_reply":
                await answer_callback(cb_id, "Cancelled")
                clear_reply_state(cid)
                clear_state(cid)
                await edit_message(cid, mid, "✅ Cancelled.")
                return JSONResponse({"ok": True})

            if cb_data == "close_settings":
                await answer_callback(cb_id)
                await edit_message(cid, mid, "✅ Settings closed.")
                return JSONResponse({"ok": True})

            if cb_data == "back_settings":
                await answer_callback(cb_id)
                kb = admin_settings_keyboard() if is_admin(cid) else user_settings_keyboard()
                await edit_message(cid, mid, "⚙️ <b>Settings</b>", parse_mode="HTML", reply_markup=kb)
                return JSONResponse({"ok": True})

            if cb_data == "developer_credits":
                await answer_callback(cb_id)
                credit_message = get_credit_message()
                if is_admin(cid):
                    await edit_message(
                        cid,
                        mid,
                        f"🛠 <b>Developer &amp; Credits</b>\n\n{escape_html(credit_message)}",
                        parse_mode="HTML",
                        reply_markup=ikb([[btn("➕ Add New Message", "set_credit_message")], [btn("🔙 Back", "back_settings")]]),
                    )
                else:
                    await edit_message(
                        cid,
                        mid,
                        f"🛠 <b>Developer &amp; Credits</b>\n\n{escape_html(credit_message)}",
                        parse_mode="HTML",
                        reply_markup=ikb([[btn("🔙 Back", "back_settings")]]),
                    )
                return JSONResponse({"ok": True})

            if cb_data == "set_credit_message":
                if not is_admin(cid):
                    await answer_callback(cb_id, "Unauthorized")
                    return JSONResponse({"ok": True})
                await answer_callback(cb_id)
                set_state(cid, "awaiting_credit_message")
                await send_message(cid, "✍️ Send the new developer and credits message:", reply_markup=ikb([[btn("❌ Cancel", "cancel_reply")]]))
                return JSONResponse({"ok": True})

            if cb_data == "set_system":
                await answer_callback(cb_id)
                current = get_user_system(cid)
                info = f"Current: <i>{escape_html(current[:200])}</i>" if current else "No custom instructions set."
                set_state(cid, "awaiting_system_instructions")
                await send_message(
                    cid,
                    f"🧠 <b>System Instructions</b>\n\n{info}\n\nType new instructions or send /clear_system to remove:",
                    parse_mode="HTML",
                    reply_markup=ikb([
                        [btn("🗑️ Clear Instructions", "clear_system")],
                        [btn("🔙 Back", "back_settings")],
                    ]),
                )
                return JSONResponse({"ok": True})

            if cb_data == "clear_system":
                await answer_callback(cb_id, "Cleared!")
                clear_user_system(cid)
                clear_state(cid)
                await edit_message(cid, mid, "🗑️ System instructions cleared.", parse_mode="HTML")
                return JSONResponse({"ok": True})

            if cb_data == "set_voice":
                await answer_callback(cb_id)
                from tts import fetch_microsoft_voices
                voices = await fetch_microsoft_voices()
                current_voice = get_user_voice(cid)
                voice_count = len(voices)
                await edit_message(
                    cid, mid,
                    f"🎙️ <b>TTS Voice Selection</b>\n\nAvailable neural voices: {voice_count}\nCurrent voice: <code>{current_voice}</code>\n\nSelect a voice (sorted by language-voice-female):",
                    parse_mode="HTML",
                    reply_markup=voice_keyboard(0),
                )
                return JSONResponse({"ok": True})

            if cb_data.startswith("voice_page:"):
                page = int(cb_data.split(":", 1)[1])
                await answer_callback(cb_id)
                current_voice = get_user_voice(cid)
                from tts import MICROSOFT_VOICES_CACHE
                voice_count = len(MICROSOFT_VOICES_CACHE)
                await edit_message(
                    cid,
                    mid,
                    f"🎙️ <b>TTS Voice Selection</b>\n\nAvailable neural voices: {voice_count}\nCurrent voice: <code>{current_voice}</code>\n\nSelect a voice (sorted by language-voice-female):",
                    parse_mode="HTML",
                    reply_markup=voice_keyboard(page),
                )
                return JSONResponse({"ok": True})

            if cb_data.startswith("voice:"):
                voice_name = cb_data.split(":", 1)[1]
                set_user_voice(cid, voice_name)
                await answer_callback(cb_id, f"Voice set: {voice_name}")
                await edit_message(
                    cid,
                    mid,
                    f"✅ TTS voice changed to <code>{voice_name}</code>",
                    parse_mode="HTML",
                    reply_markup=ikb([[btn("🔙 Back", "back_settings")]]),
                )
                return JSONResponse({"ok": True})

            if cb_data == "set_temp":
                await answer_callback(cb_id)
                current_temp = get_user_temp(cid)
                await edit_message(
                    cid, mid,
                    f"🌡️ <b>Temperature Setting</b>\n\nCurrent: <code>{current_temp}</code>\n\nHigher = more creative, Lower = more precise:",
                    parse_mode="HTML",
                    reply_markup=temp_keyboard(),
                )
                return JSONResponse({"ok": True})

            if cb_data.startswith("temp:"):
                temp_val = float(cb_data.split(":")[1])
                set_user_temp(cid, temp_val)
                await answer_callback(cb_id, f"Temperature: {temp_val}")
                await edit_message(cid, mid, f"✅ Temperature set to <code>{temp_val}</code>", parse_mode="HTML", reply_markup=ikb([[btn("🔙 Back", "back_settings")]]))
                return JSONResponse({"ok": True})

            if cb_data == "set_model":
                await answer_callback(cb_id)
                current_model = get_user_model(cid)
                await edit_message(
                    cid, mid,
                    f"🤖 <b>AI Model Selection</b>\n\nCurrent model: <code>{current_model}</code>\n\nSelect a model:",
                    parse_mode="HTML",
                    reply_markup=model_keyboard(),
                )
                return JSONResponse({"ok": True})

            if cb_data.startswith("model:"):
                model_name = cb_data.split(":", 1)[1]
                set_user_model(cid, model_name)
                await answer_callback(cb_id, f"Model set: {model_name}")
                await edit_message(
                    cid,
                    mid,
                    f"✅ AI model changed to <code>{model_name}</code>",
                    parse_mode="HTML",
                    reply_markup=ikb([[btn("🔙 Back", "back_settings")]]),
                )
                return JSONResponse({"ok": True})

            if cb_data.startswith("reply_admin:"):
                await answer_callback(cb_id)
                set_reply_state(cid, ADMINS[0])
                await send_message(cid, "✍️ Reply to admin with text, voice, or any attachment.", reply_markup=ikb([[btn("❌ Cancel", "cancel_reply")]]))
                return JSONResponse({"ok": True})

            if cb_data.startswith("reply_user:"):
                target = int(cb_data.split(":")[1])
                await answer_callback(cb_id)
                set_reply_state(cid, target)
                target_name = get_username(target)
                await send_message(cid, f"✍️ Message <b>{escape_html(target_name)}</b> (<code>{target}</code>). Send text, voice, or any attachment:", parse_mode="HTML", reply_markup=ikb([[btn("❌ Cancel", "cancel_reply")]]))
                return JSONResponse({"ok": True})

            if cb_data.startswith("regen_img:"):
                prompt = cb_data.split(":", 1)[1]
                await answer_callback(cb_id, "Regenerating...")
                ensure_user(cid, name)
                await execute_image(cid, prompt, name)
                return JSONResponse({"ok": True})

            if cb_data == "admin_total":
                if not is_admin(cid):
                    await answer_callback(cb_id, "Unauthorized")
                    return JSONResponse({"ok": True})
                await answer_callback(cb_id)
                users = get_all_users()
                if not users:
                    await send_message(cid, "No users registered.", reply_markup=ikb([[btn("🔙 Back", "back_settings")]]))
                    return JSONResponse({"ok": True})
                rows = []
                for uid_str, uname in users.items():
                    uid_int = int(uid_str)
                    if uid_int not in ADMINS:
                        rows.append([btn(f"💬 Message {uname}", f"reply_user:{uid_str}"), btn(f"🚫 Ban", f"ban_confirm:{uid_str}")])
                text = f"📊 <b>Total Users: {len(users)}</b>\n\n"
                text += "".join(f"🆔 <code>{u}</code> — {escape_html(n)}\n" for u, n in users.items())
                rows.append([btn("🔙 Back", "back_settings")])
                await send_message(cid, text, parse_mode="HTML", reply_markup=ikb(rows))
                return JSONResponse({"ok": True})

            if cb_data.startswith("ban_confirm:"):
                if not is_admin(cid):
                    await answer_callback(cb_id, "Unauthorized")
                    return JSONResponse({"ok": True})
                target_str = cb_data.split(":")[1]
                uname = get_all_users().get(target_str, "Unknown")
                await answer_callback(cb_id)
                await send_message(
                    cid,
                    f"⚠️ Ban <b>{escape_html(uname)}</b> (<code>{target_str}</code>)?",
                    parse_mode="HTML",
                    reply_markup=ikb([
                        [btn("✅ Yes, Ban", f"ban_yes:{target_str}"), btn("❌ Cancel", "back_settings")],
                    ]),
                )
                return JSONResponse({"ok": True})

            if cb_data.startswith("ban_yes:"):
                if not is_admin(cid):
                    await answer_callback(cb_id, "Unauthorized")
                    return JSONResponse({"ok": True})
                target = int(cb_data.split(":")[1])
                uname = get_all_users().get(str(target), "Unknown")
                ban_user(target, uname)
                await answer_callback(cb_id, "Banned!")
                await edit_message(cid, mid, f"🚫 <b>{escape_html(uname)}</b> (<code>{target}</code>) banned.", parse_mode="HTML")
                await send_message(target, "🚫 You have been banned from using this bot.")
                return JSONResponse({"ok": True})

            if cb_data == "admin_banned":
                if not is_admin(cid):
                    await answer_callback(cb_id, "Unauthorized")
                    return JSONResponse({"ok": True})
                await answer_callback(cb_id)
                banned = get_banned_users()
                if not banned:
                    await send_message(cid, "✅ No banned users.", reply_markup=ikb([[btn("🔙 Back", "back_settings")]]))
                    return JSONResponse({"ok": True})
                rows = []
                for uid_str, uname in banned.items():
                    rows.append([btn(f"✅ Unban {uname} ({uid_str})", f"do_unban:{uid_str}")])
                text = f"🚫 <b>Banned Users: {len(banned)}</b>\n\n"
                text += "".join(f"🆔 <code>{u}</code> — {escape_html(n)}\n" for u, n in banned.items())
                rows.append([btn("🔙 Back", "back_settings")])
                await send_message(cid, text, parse_mode="HTML", reply_markup=ikb(rows))
                return JSONResponse({"ok": True})

            if cb_data == "admin_broadcast":
                if not is_admin(cid):
                    await answer_callback(cb_id, "Unauthorized")
                    return JSONResponse({"ok": True})
                await answer_callback(cb_id)
                set_state(cid, "awaiting_broadcast")
                await send_message(cid, "📢 Send your broadcast as text, voice, photo, document, audio, video, animation, or sticker.", reply_markup=ikb([[btn("❌ Cancel", "cancel_reply")]]))
                return JSONResponse({"ok": True})

            if cb_data.startswith("broadcast_clear_failed:"):
                if not is_admin(cid):
                    await answer_callback(cb_id, "Unauthorized")
                    return JSONResponse({"ok": True})
                await answer_callback(cb_id)
                failed_users = _get_broadcast_failed(cid)
                if not failed_users:
                    await send_message(cid, "✅ No failed users stored.")
                    return JSONResponse({"ok": True})
                for uid in failed_users:
                    remove_all_user_data(uid)
                clear_state(cid)
                await send_message(cid, f"🧹 Cleared data for {len(failed_users)} failed users.")
                return JSONResponse({"ok": True})

            await answer_callback(cb_id, "Unknown action.")
            return JSONResponse({"ok": True})

        if "message" not in data:
            return JSONResponse({"ok": True})

        message = data["message"]
        cid = message["chat"]["id"]
        name = get_user_name(message)

        group_prompt = None
        if is_group_chat(message):
            group_prompt = extract_group_prompt(message)
            if group_prompt is None:
                return JSONResponse({"ok": True})
            if "text" in message:
                message["text"] = group_prompt
            elif "caption" in message:
                message["caption"] = group_prompt
            else:
                message["text"] = group_prompt

        if check_banned(cid):
            await send_banned_message(cid)
            return JSONResponse({"ok": True})

        st = get_state(cid)
        if st == "awaiting_broadcast" and is_admin(cid) and "text" not in message:
            clear_state(cid)
            failed_ids = await run_broadcast_copy(cid, cid, message["message_id"])
            if failed_ids:
                _set_broadcast_failed(cid, failed_ids)
            return JSONResponse({"ok": True})

        if st and ("text" in message or "caption" in message):
            text = (message.get("text") or message.get("caption") or "").strip()

            if text.startswith("/"):
                clear_state(cid)
            else:
                if st == "awaiting_system_instructions":
                    clear_state(cid)
                    set_user_system(cid, text)
                    await send_message(
                        cid,
                        f"✅ System instructions updated:\n\n<i>{escape_html(text[:500])}</i>",
                        parse_mode="HTML",
                        reply_markup=ikb([[btn("🔙 Back", "back_settings")]]),
                    )
                    return JSONResponse({"ok": True})

                if st == "awaiting_broadcast":
                    clear_state(cid)
                    if not is_admin(cid):
                        return JSONResponse({"ok": True})
                    failed_ids = await run_broadcast(cid, text=text)
                    if failed_ids:
                        _set_broadcast_failed(cid, failed_ids)
                    return JSONResponse({"ok": True})

                if st == "awaiting_credit_message":
                    clear_state(cid)
                    if not is_admin(cid):
                        return JSONResponse({"ok": True})
                    set_credit_message(text)
                    await send_message(cid, "✅ Developer and credits message updated.", reply_markup=ikb([[btn("🔙 Back", "back_settings")]]))
                    return JSONResponse({"ok": True})
                if st == "awaiting_memory_add":
                    clear_state(cid)
                    save_memory(cid, text)
                    await send_message(cid, "✅ Memory saved.")
                    return JSONResponse({"ok": True})

                if st.startswith("tool:"):
                    if st == "tool:text_refiner":
                        await run_text_refiner(cid, text)
                        return JSONResponse({"ok": True})
                    if st == "tool:text_translator:text":
                        set_state(cid, f"tool:text_translator:lang:{text}")
                        await send_message(cid, "Send your target language for translation function.", reply_markup=TOOL_CANCEL)
                        return JSONResponse({"ok": True})
                    if st.startswith("tool:text_translator:lang:"):
                        source_text = st.split(":", 3)[3]
                        lang_code, lang_name = resolve_language(text)
                        if not lang_code or not lang_name:
                            await send_message(cid, "❌ Invalid language. Send a valid language code or name.", reply_markup=TOOL_CANCEL)
                            return JSONResponse({"ok": True})
                        set_state(cid, "tool:text_translator:text")
                        await run_text_translator(cid, source_text, lang_code, lang_name)
                        return JSONResponse({"ok": True})
                    if st == "tool:pdf_creator":
                        await run_pdf_creator(cid, text)
                        return JSONResponse({"ok": True})
                    if st == "tool:text_analyzer":
                        await run_text_analyzer(cid, text)
                        return JSONResponse({"ok": True})
                    if st == "tool:audio_transcriber":
                        await send_message(cid, "Upload voice/audio file to transcribe.", reply_markup=TOOL_CANCEL)
                        return JSONResponse({"ok": True})

                if st.startswith("awaiting_file_prompt:"):
                    clear_state(cid)
                    ensure_user(cid, name)
                    await send_chat_action(cid, "typing")
                    fd = get_file_data(cid)
                    if not fd:
                        await send_message(cid, "❌ File not found. Please upload again.")
                        return JSONResponse({"ok": True})
                    file_display = st.split(":", 1)[1] if ":" in st else "file"
                    save_message(cid, "user", f"[File: {file_display}] {text}")
                    parts: list = [{"text": text}]
                    if fd.get("base64"):
                        parts.append({"inlineData": {"mimeType": fd["mime_type"], "data": fd["base64"]}})
                    await handle_gemini(
                        cid,
                        parts,
                        get_system_text(name, cid),
                        use_tools=False,
                    )
                    return JSONResponse({"ok": True})

        if st and message.get("document") and st.startswith("tool:"):
            doc = message["document"]
            file_name = (doc.get("file_name") or "").lower()
            if not file_name.endswith(".txt"):
                await send_message(cid, "❌ Please upload a .txt file.", reply_markup=TOOL_CANCEL)
                return JSONResponse({"ok": True})
            file_bytes = await download_telegram_file(doc["file_id"])
            if not file_bytes:
                await send_message(cid, "❌ Failed to download .txt file.", reply_markup=TOOL_CANCEL)
                return JSONResponse({"ok": True})
            limit = None if st == "tool:text_analyzer" else MAX_TOOL_TEXT_FILE_BYTES
            tool_text, error = parse_text_document_bytes(file_bytes, limit_bytes=limit)
            if error:
                await send_message(cid, error, reply_markup=TOOL_CANCEL)
                return JSONResponse({"ok": True})
            if not tool_text:
                await send_message(cid, "❌ Could not read text from this file.", reply_markup=TOOL_CANCEL)
                return JSONResponse({"ok": True})

            if st == "tool:text_refiner":
                await run_text_refiner(cid, tool_text)
                return JSONResponse({"ok": True})
            if st == "tool:text_translator:text":
                set_state(cid, f"tool:text_translator:lang:{tool_text}")
                await send_message(cid, "Send your target language for translation function.", reply_markup=TOOL_CANCEL)
                return JSONResponse({"ok": True})
            if st == "tool:text_analyzer":
                await run_text_analyzer(cid, tool_text)
                return JSONResponse({"ok": True})

        reply_target = get_reply_state(cid)
        if reply_target is not None and message.get("voice"):
            clear_reply_state(cid)
            if reply_target == -1:
                voice_data = await download_telegram_file(message["voice"]["file_id"])
                if not voice_data:
                    await send_message(cid, "❌ Failed to send voice feedback.")
                    return JSONResponse({"ok": True})
                for admin_id in ADMINS:
                    await send_voice_bytes(admin_id, voice_data, None, "feedback.ogg", message["voice"].get("mime_type", "audio/ogg"))
                    await send_message(admin_id, f"📬 <b>Voice Feedback</b>\n👤 {escape_html(name)}\n🆔 <code>{cid}</code>", parse_mode="HTML", reply_markup=admin_user_reply_keyboard(cid, name))
                await send_message(cid, "✅ Voice feedback sent!")
                return JSONResponse({"ok": True})
            await relay_voice_reply(cid, name, reply_target, message["voice"], is_admin(cid))
            return JSONResponse({"ok": True})

        attachment_keys = ("photo", "document", "audio", "video", "video_note", "animation", "sticker")
        if reply_target is not None and any(message.get(key) for key in attachment_keys):
            clear_reply_state(cid)
            if reply_target == -1:
                for admin_id in ADMINS:
                    await copy_message(admin_id, cid, message["message_id"], reply_markup=admin_user_reply_keyboard(cid, name))
                    await send_message(admin_id, f"📬 <b>Attachment Feedback</b>\n👤 {escape_html(name)}\n🆔 <code>{cid}</code>", parse_mode="HTML")
                await send_message(cid, "✅ Attachment feedback sent!")
                return JSONResponse({"ok": True})

            if is_admin(cid):
                result = await copy_message(reply_target, cid, message["message_id"], reply_markup=admin_reply_keyboard(reply_target))
                status = "✅ Sent" if result and result.get("ok") else "❌ Failed"
                await send_message(cid, f"{status} attachment to <code>{reply_target}</code>.", parse_mode="HTML")
            else:
                for admin_id in ADMINS:
                    await copy_message(admin_id, cid, message["message_id"], reply_markup=admin_user_reply_keyboard(cid, name))
                    await send_message(admin_id, f"📩 <b>Attachment reply from user</b>\n\n👤 {escape_html(name)}\n🆔 <code>{cid}</code>", parse_mode="HTML")
                await send_message(cid, "✅ Attachment reply sent!")
            return JSONResponse({"ok": True})

        if reply_target is not None and "text" in message:
            clear_reply_state(cid)
            reply_text = message["text"].strip()

            if reply_target == -1:
                await send_feedback_to_admins(cid, name, reply_text)
                await send_message(cid, "✅ Feedback sent!")
                return JSONResponse({"ok": True})

            if is_admin(cid):
                admin_msg = f"📩 <b>Message from Admin:</b>\n\n{escape_html(reply_text)}"
                result = await send_message(reply_target, admin_msg, parse_mode="HTML", reply_markup=admin_reply_keyboard(reply_target))
                status = "✅ Sent" if result and result.get("ok") else "❌ Failed"
                await send_message(cid, f"{status} to <code>{reply_target}</code>.", parse_mode="HTML")
            else:
                user_msg = (
                    f"📩 <b>Reply from User</b>\n\n"
                    f"👤 {escape_html(name)}\n"
                    f"🆔 <code>{cid}</code>\n\n"
                    f"💬 {escape_html(reply_text)}"
                )
                for admin_id in ADMINS:
                    await send_message(admin_id, user_msg, parse_mode="HTML", reply_markup=admin_user_reply_keyboard(cid, name))
                await send_message(cid, "✅ Reply sent!")
            return JSONResponse({"ok": True})

        if message.get("photo"):
            await handle_photo(cid, message, name)
            return JSONResponse({"ok": True})

        if message.get("voice"):
            if st == "tool:audio_transcriber":
                await transcribe_from_telegram_message(cid, message)
                return JSONResponse({"ok": True})
            ensure_user(cid, name)
            await handle_voice(cid, message["voice"], name)
            return JSONResponse({"ok": True})

        if message.get("document"):
            if st == "tool:audio_transcriber":
                await transcribe_from_telegram_message(cid, message)
                return JSONResponse({"ok": True})
            await handle_document(cid, message, name)
            return JSONResponse({"ok": True})

        if message.get("audio"):
            if st == "tool:audio_transcriber":
                await transcribe_from_telegram_message(cid, message)
                return JSONResponse({"ok": True})
            await handle_audio(cid, message, name)
            return JSONResponse({"ok": True})

        if message.get("video") or message.get("video_note"):
            await handle_video(cid, message, name)
            return JSONResponse({"ok": True})

        if message.get("animation"):
            await handle_animation(cid, message, name)
            return JSONResponse({"ok": True})

        if message.get("sticker"):
            await handle_sticker(cid, message, name)
            return JSONResponse({"ok": True})

        if "text" not in message:
            await send_message(cid, "⚠️ Unsupported message type. Send text, images, voice, documents, audio, or video.")
            return JSONResponse({"ok": True})

        text = message["text"]

        if text == "/start":
            if user_exists(cid):
                remove_all_user_data(cid)
            save_user(cid, name)
            clear_history(cid)
            welcome = (
                f"👋 <b>Hi {escape_html(name)}, Welcome to Sahana AI!</b>\n\n"
                f"🤖 <b>Sahana AI</b> is your intelligent AI assistant, optimized with advanced large language models "
                f"to deliver fast, accurate, and context-aware responses.\n\n"
                f"<b>Here's what Sahana AI can do for you:</b>\n\n"
                f"💬 <b>Natural Conversations</b> — Chat naturally on any topic\n"
                f"🌐 <b>Real-time Web Search</b> — Get the most up-to-date information\n"
                f"🎬 <b>YouTube Analysis</b> — Transcribe, summarize &amp; analyze videos\n"
                f"🧾 <b>Text to PDF</b> — Create downloadable PDF files from your prompts\n"
                f"🎨 <b>Image Generation</b> — Create stunning AI-generated images\n"
                f"🖼️ <b>Image Analysis</b> — Upload a photo and get detailed descriptions\n"
                f"🔗 <b>URL Browsing</b> — Browse and analyze any public webpage\n"
                f"📄 <b>Document Processing</b> — Analyze PDFs, DOCX, spreadsheets &amp; more\n"
                f"🎵 <b>Audio &amp; Video</b> — Process audio and video files\n"
                f"💻 <b>Code in 100+ Languages</b> — Write, debug &amp; explain code\n"
                f"🌍 <b>Translation</b> — Translate between languages effortlessly\n"
                f"📊 <b>Math &amp; Science</b> — Solve complex problems step by step\n"
                f"📖 <b>Summarization</b> — Condense long texts into key points\n"
                f"🧠 <b>Memory</b> — Sahana AI remembers your conversations for contextual responses\n"
                f"📋 <b>Custom Instructions</b> — Set system instructions for personalized behavior\n"
                f"🎙️ <b>Voice Responses</b> — Sahana AI can reply with voice in many languages\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Sahana AI is <b>fast, free, and powerful</b>. With its agentic workflow, "
                f"it understands your intent and routes your requests intelligently.\n\n"
                f"🙏 <i>Thank you for using Sahana AI! If you love it, share it with your friends.</i>"
            )
            await send_message(cid, welcome, parse_mode="HTML", reply_markup=start_keyboard())
            await send_message(
                cid,
                "✍️ <b>Type or speak your prompt — I'm ready to dive in!</b>\n\n"
                "💡 Here are some template prompts to get you started:",
                parse_mode="HTML",
                reply_markup=template_prompts_keyboard(),
            )
            return JSONResponse({"ok": True})

        if text in ("/settings", "/menu"):
            ensure_user(cid, name)
            kb = admin_settings_keyboard() if is_admin(cid) else user_settings_keyboard()
            await send_message(cid, "⚙️ <b>Settings</b>", parse_mode="HTML", reply_markup=kb)
            return JSONResponse({"ok": True})

        if text == "/tools":
            ensure_user(cid, name)
            set_state(cid, "tool:menu")
            await open_tools_menu(cid)
            return JSONResponse({"ok": True})

        if text == "/clear":
            ensure_user(cid, name)
            clear_history(cid)
            clear_memories(cid)
            await send_message(cid, "🗑️ Conversation and memories cleared.")
            return JSONResponse({"ok": True})

        if text == "/cls":
            ensure_user(cid, name)
            clear_file_data(cid)
            await send_message(cid, "🧹 Last attachment cleared.")
            return JSONResponse({"ok": True})

        if text == "/exit":
            remove_all_user_data(cid)
            await send_message(
                cid,
                "👋 <b>All your data has been cleared.</b>\n\nSend /start to begin fresh.",
                parse_mode="HTML",
            )
            return JSONResponse({"ok": True})

        if text == "/total":
            if not is_admin(cid):
                await send_message(cid, "Command not recognized.")
                return JSONResponse({"ok": True})
            users = get_all_users()
            response_text = f"📊 <b>Total Users: {len(users)}</b>\n\n"
            response_text += "".join(f"🆔 <code>{u}</code> — {escape_html(n)}\n" for u, n in users.items()) or "No users."
            await send_message(cid, response_text, parse_mode="HTML")
            return JSONResponse({"ok": True})

        if text.startswith("/sendMessage"):
            if not is_admin(cid):
                await send_message(cid, "Command not recognized.")
                return JSONResponse({"ok": True})
            content = text.replace("/sendMessage", "", 1).strip()
            if " - " not in content:
                await send_message(cid, "Format: /sendMessage &lt;user_id&gt; - &lt;message&gt;", parse_mode="HTML")
                return JSONResponse({"ok": True})
            target_str, msg_content = content.split(" - ", 1)
            target_str = target_str.strip()
            if not target_str.lstrip("-").isdigit():
                await send_message(cid, "Invalid user ID.")
                return JSONResponse({"ok": True})
            target = int(target_str)
            if not user_exists(target):
                await send_message(cid, f"User <code>{target}</code> not found.", parse_mode="HTML")
                return JSONResponse({"ok": True})
            admin_msg = f"📩 <b>Message from Admin:</b>\n\n{escape_html(msg_content.strip())}"
            result = await send_message(target, admin_msg, parse_mode="HTML", reply_markup=admin_reply_keyboard(target))
            status = "✅ Sent" if result and result.get("ok") else "❌ Failed"
            await send_message(cid, f"{status} to <code>{target}</code>.", parse_mode="HTML")
            return JSONResponse({"ok": True})

        if text.startswith("/broadcast"):
            if not is_admin(cid):
                await send_message(cid, "Command not recognized.")
                return JSONResponse({"ok": True})
            broadcast_msg = text.replace("/broadcast", "", 1).strip()
            if not broadcast_msg:
                await send_message(cid, "Provide a message after /broadcast.")
                return JSONResponse({"ok": True})
            failed_ids = await run_broadcast(cid, text=broadcast_msg)
            if failed_ids:
                _set_broadcast_failed(cid, failed_ids)
            return JSONResponse({"ok": True})

        if text.startswith("/ban"):
            if not is_admin(cid):
                await send_message(cid, "Command not recognized.")
                return JSONResponse({"ok": True})
            target_str = text.replace("/ban", "", 1).strip()
            if not target_str.lstrip("-").isdigit():
                await send_message(cid, "Format: /ban &lt;user_id&gt;", parse_mode="HTML")
                return JSONResponse({"ok": True})
            target = int(target_str)
            uname = get_all_users().get(str(target), "Unknown")
            ban_user(target, uname)
            await send_message(cid, f"🚫 Banned <code>{target}</code> ({escape_html(uname)})", parse_mode="HTML")
            await send_message(target, "🚫 You have been banned from using this bot.")
            return JSONResponse({"ok": True})

        if text.startswith("/unban"):
            if not is_admin(cid):
                await send_message(cid, "Command not recognized.")
                return JSONResponse({"ok": True})
            target_str = text.replace("/unban", "", 1).strip()
            if not target_str.lstrip("-").isdigit():
                await send_message(cid, "Format: /unban &lt;user_id&gt;", parse_mode="HTML")
                return JSONResponse({"ok": True})
            target = int(target_str)
            unban_user(target)
            await send_message(cid, f"✅ Unbanned <code>{target}</code>", parse_mode="HTML")
            await send_message(target, "🎉 You have been unbanned! Send /start to continue.")
            return JSONResponse({"ok": True})

        if text.startswith("/feedback"):
            ensure_user(cid, name)
            feedback_text = text.replace("/feedback", "", 1).strip()
            if not feedback_text:
                set_reply_state(cid, -1)
                await send_message(cid, "💬 Send feedback as text, voice, photo, document, audio, video, animation, or sticker.", reply_markup=ikb([[btn("❌ Cancel", "cancel_reply")]]))
                return JSONResponse({"ok": True})
            await send_feedback_to_admins(cid, name, feedback_text)
            await send_message(cid, "✅ Feedback sent!")
            return JSONResponse({"ok": True})

        if text == "/clear_system":
            ensure_user(cid, name)
            clear_user_system(cid)
            await send_message(cid, "🗑️ System instructions cleared.")
            return JSONResponse({"ok": True})

        if text == "/help":
            ensure_user(cid, name)
            if is_admin(cid):
                help_text = (
                    "📖 <b>Sahana AI Admin Help</b>\n\n"
                    "<b>Admin Commands</b>\n"
                    "/total — View all users\n"
                    "/sendMessage &lt;id&gt; - &lt;text&gt; — Message a user\n"
                    "/broadcast &lt;text&gt; — Broadcast text\n"
                    "/ban &lt;id&gt; — Ban user\n"
                    "/unban &lt;id&gt; — Unban user\n\n"
                    "<b>General Commands</b>\n"
                    "/start — Restart and reset your session\n"
                    "/settings — Open admin settings\n"
                    "/clear — Clear chat history\n"
                    "/memory add &lt;text&gt; — Save memory manually\n"
                    "/memory list — View saved memories\n"
                    "/memory clear — Clear saved memories\n"
                    "/cls — Clear stored attachment\n"
                    "/clear_system — Remove custom system instructions\n"
                    "/tools — Open productivity tools\n/help — Show admin help\n"
                )
            else:
                help_text = (
                    "📖 <b>Sahana AI User Help</b>\n\n"
                    "<b>User Commands</b>\n"
                    "/start — Restart and reset your session\n"
                    "/settings — Open settings\n"
                    "/clear — Clear chat history\n"
                    "/memory add &lt;text&gt; — Save memory manually\n"
                    "/memory list — View saved memories\n"
                    "/memory clear — Clear saved memories\n"
                    "/cls — Clear stored attachment\n"
                    "/feedback — Send feedback (text, voice, or attachments)\n"
                    "/clear_system — Remove custom system instructions\n"
                    "/tools — Open productivity tools\n/help — Show this help\n\n"
                    "<b>Supported Inputs</b>\n"
                    "• Text prompts and coding requests\n"
                    "• Documents and code files (HTML, Markdown, PDF, DOCX, XLSX, TXT, etc.)\n"
                    "• Audio and voice files\n"
                    "• Videos and animations\n"
                    "• Images and stickers\n"
                )
            await send_message(cid, help_text, parse_mode="HTML")
            return JSONResponse({"ok": True})

        if text.startswith("/"):
            if text.startswith("/memory"):
                ensure_user(cid, name)
                payload = text.replace("/memory", "", 1).strip()
                if payload.startswith("add "):
                    save_memory(cid, payload[4:].strip())
                    await send_message(cid, "✅ Memory saved.")
                elif payload == "list":
                    memories = get_memories(cid)
                    msg = "\n".join(f"{i+1}. {m}" for i, m in enumerate(memories)) if memories else "No memories saved."
                    await send_message(cid, msg)
                elif payload == "clear":
                    clear_memories(cid)
                    await send_message(cid, "✅ Memories cleared.")
                else:
                    await send_message(cid, "Usage: /memory add <text> | /memory list | /memory clear")
                return JSONResponse({"ok": True})
            await send_message(cid, "Command not recognized. Use /help to see available commands.")
            return JSONResponse({"ok": True})

        if not user_exists(cid):
            save_user(cid, name)
            welcome = (
                f"👋 <b>Hi {escape_html(name)}, Welcome to Sahana AI!</b>\n\n"
                f"Send /start for the full introduction, or just keep chatting!"
            )
            await send_message(cid, welcome, parse_mode="HTML", reply_markup=start_keyboard())

        ensure_user(cid, name)
        await send_chat_action(cid, "typing")
        prompt_text = text.strip()
        save_message(cid, "user", prompt_text)
        current_parts = [{"text": prompt_text}]
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
        return JSONResponse({"ok": True})

    except Exception:
        logger.exception("webhook_failed")
        return JSONResponse({"ok": True})
