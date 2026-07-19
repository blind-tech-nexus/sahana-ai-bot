import urllib.parse
from config import TEMPLATE_PROMPTS, SHARE_TEXT
from languages import LANGUAGES, LANGUAGE_NAME_BY_CODE


def btn(text: str, callback_data: str) -> dict:
    return {"text": text, "callback_data": callback_data}


def url_btn(text: str, url: str) -> dict:
    return {"text": text, "url": url}


def ikb(rows: list[list[dict]]) -> dict:
    return {"inline_keyboard": rows}


def start_keyboard() -> dict:
    return ikb([
        [btn("⚙️ Settings", "open_settings"), btn("📤 Share Bot", "share_bot")],
    ])


def template_prompts_keyboard() -> dict:
    rows = []
    for i in range(0, len(TEMPLATE_PROMPTS), 2):
        row = [btn(f"💡 {TEMPLATE_PROMPTS[i][:30]}", f"tp:{i}")]
        if i + 1 < len(TEMPLATE_PROMPTS):
            row.append(btn(f"💡 {TEMPLATE_PROMPTS[i+1][:30]}", f"tp:{i+1}"))
        rows.append(row)
    return ikb(rows)


def user_settings_keyboard() -> dict:
    return ikb([
        [btn("🧠 System Instructions", "set_system"), btn("🎙️ TTS Voice", "set_voice")],
        [btn("🌡️ Temperature", "set_temp"), btn("🤖 AI Model", "set_model")],
        [btn("🧠 Memory", "memory_settings")],
        [btn("🔧 Tools & Features", "open_tools")],
        [btn("🗑️ Clear Chat", "clear"), btn("🧹 Clear Attachment", "cls")],
        [btn("💬 Feedback", "feedback_prompt")],
        [btn("🔄 Export Chat", "export_chat")],
        [btn("🛠 Developer & Credits", "developer_credits")],
        [btn("❌ Close", "close_settings")],
    ])


def admin_settings_keyboard() -> dict:
    return ikb([
        [btn("🧠 System Instructions", "set_system"), btn("🎙️ TTS Voice", "set_voice")],
        [btn("🌡️ Temperature", "set_temp"), btn("🤖 AI Model", "set_model")],
        [btn("🧠 Memory", "memory_settings")],
        [btn("🔧 Tools & Features", "open_tools")],
        [btn("🗑️ Clear Chat", "clear"), btn("🧹 Clear Attachment", "cls")],
        [btn("📊 Total Users", "admin_total"), btn("🚫 Banned Users", "admin_banned")],
        [btn("📢 Broadcast", "admin_broadcast")],
        [btn("🔄 Export Chat", "export_chat")],
        [btn("🛠 Developer & Credits", "developer_credits")],
        [btn("⚠️ Clear Full Data", "admin_clear_full_data")],
        [btn("❌ Close", "close_settings")],
    ])


def language_name(code: str) -> str:
    return LANGUAGE_NAME_BY_CODE.get(code, code)


def voice_button_text(voice: dict) -> str:
    """Generate button text for a voice in format: Language - Voice Name - Gender."""
    lang = voice.get("language", "Unknown")
    name = voice.get("name", "Unknown")
    gender = voice.get("gender", "Unknown")
    # Extract just the locale part from language if it contains more details
    if " - " in lang:
        lang = lang.split(" - ")[0]
    return f"{lang} - {name} - {gender}"


def voice_keyboard(page: int = 0, per_page: int = 20) -> dict:
    """Generate keyboard for Microsoft TTS voices with pagination."""
    from tts import MICROSOFT_VOICES_CACHE
    
    voices = MICROSOFT_VOICES_CACHE if MICROSOFT_VOICES_CACHE else []
    total = len(voices)
    last_page = max(0, (total - 1) // per_page) if total > 0 else 0
    page = min(max(page, 0), last_page)
    start = page * per_page
    items = voices[start:start + per_page]
    rows: list[list[dict]] = []

    for i in range(0, len(items), 2):
        row = [btn(voice_button_text(items[i]), f"voice:{items[i]['name']}")]
        if i + 1 < len(items):
            row.append(btn(voice_button_text(items[i + 1]), f"voice:{items[i + 1]['name']}"))
        rows.append(row)

    nav: list[dict] = []
    if page > 0:
        nav.append(btn("⬅️ Prev", f"voice_page:{page - 1}"))
    if page < last_page:
        nav.append(btn("Next ➡️", f"voice_page:{page + 1}"))
    if nav:
        rows.append(nav)

    rows.append([btn("🔙 Back", "back_settings")])
    return ikb(rows)


def temp_keyboard() -> dict:
    return ikb([
        [btn("🧊 0.0 Precise", "temp:0.0"), btn("❄️ 0.3 Balanced", "temp:0.3")],
        [btn("🌤️ 0.7 Creative", "temp:0.7"), btn("🔥 1.0 Very Creative", "temp:1.0")],
        [btn("🌋 1.5 Wild", "temp:1.5"), btn("💥 2.0 Maximum", "temp:2.0")],
        [btn("🔙 Back", "back_settings")],
    ])


def model_keyboard() -> dict:
    return ikb([
        [btn("⚡ Sahana-1 (Fast)", "model:sahana-1"), btn("🧠 Sahana-2 (Balanced)", "model:sahana-2")],
        [btn("🔬 Sahana-3 (Advanced)", "model:sahana-3")],
        [btn("🔙 Back", "back_settings")],
    ])


def photo_keyboard() -> dict:
    return ikb([
        [btn("📝 Describe", "describe_photo")],
        [btn("❌ Cancel", "cancel_attachment")],
    ])


def file_prompt_keyboard() -> dict:
    return ikb([
        [btn("❌ Cancel", "cancel_attachment")],
    ])


def admin_reply_keyboard(uid: int) -> dict:
    return ikb([[btn("↩️ Reply Admin", f"reply_admin:{uid}")]])


def admin_user_reply_keyboard(target: int, username: str = "User") -> dict:
    return ikb([[btn(f"↩️ Message {username}", f"reply_user:{target}")]])


def broadcast_reply_keyboard() -> dict:
    return ikb([[btn("💬 Reply to Admin", "feedback_prompt")]])


def tools_keyboard() -> dict:
    return ikb([
        [btn("✨ Text refiner", "tool:text_refiner")],
        [btn("🌐 Text translator", "tool:text_translator")],
        [btn("📊 Text analyzer", "tool:text_analyzer")],
        [btn("📄 PDF creator", "tool:pdf_creator")],
        [btn("🎙️ Audio transcriber", "tool:audio_transcriber")],
        [btn("❌ Close", "tools_close")],
    ])

def share_keyboard() -> dict:
    return ikb([
        [url_btn("📤 Share Now", f"https://t.me/share/url?url=https://t.me/meroaiassistantbot_bot&text={urllib.parse.quote(SHARE_TEXT)}")],
    ])
