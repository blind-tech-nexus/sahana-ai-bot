import re
from typing import Optional
from config import BOT_USERNAME, BOT_MENTION_ALIASES

def is_group_chat(message: dict) -> bool:
    chat_type = (message.get("chat") or {}).get("type", "")
    return chat_type in {"group", "supergroup"}

def extract_group_prompt(message: dict) -> Optional[str]:
    # Only handle groups/supergroups
    chat_type = (message.get("chat") or {}).get("type", "")
    if chat_type not in {"group", "supergroup"}:
        return None

    text = (message.get("text") or message.get("caption") or "")
    text_stripped = text.strip()

    reply_to = message.get("reply_to_message")
    is_reply_to_bot = False
    if reply_to:
        from_user = reply_to.get("from") or {}
        if from_user.get("is_bot"):
            # Broad detection: any bot reply in group is considered addressed to us
            # Narrow if BOT_USERNAME is set: prefer username match but fallback to any bot
            bot_username_norm = (BOT_USERNAME or "").lower().lstrip("@")
            reply_username = (from_user.get("username") or "").lower()
            aliases_for_reply = {"sahana", "sahanai", "sahanaai", "meroaiassistantbot", "meroaiassistantbot_bot", "sahanaraiai_bot", "sahanaraiai"}
            aliases_for_reply.add(bot_username_norm)
            for a in BOT_MENTION_ALIASES:
                if a:
                    aliases_for_reply.add(a.lower().lstrip("@"))
            # If username matches known aliases or contains sahana/meroai, it's us; otherwise any bot still counts as reply to bot in group
            if not reply_username or reply_username in aliases_for_reply or "sahana" in reply_username or "meroai" in reply_username or reply_username == bot_username_norm:
                is_reply_to_bot = True
            else:
                # Be permissive: any bot reply still triggers to avoid missing due to username mismatch
                is_reply_to_bot = True

    # Build comprehensive alias set including defaults and env
    aliases = {"ai", "sahana", "sahanai", "sahanaai", "meroaiassistantbot", "meroaiassistantbot_bot", "sahanaraiai_bot", "sahanaraiai"}
    for a in BOT_MENTION_ALIASES:
        if a:
            aliases.add(a.lower().lstrip("@"))
    if BOT_USERNAME:
        bot_norm = BOT_USERNAME.lower().lstrip("@")
        aliases.add(bot_norm)
        # Also add without trailing _bot
        if bot_norm.endswith("_bot"):
            aliases.add(bot_norm[:-4])
    # Also ensure bare bot username without _bot is detectable
    # Check mention
    mentioned_alias = None
    lower_text = text.lower()
    for alias in aliases:
        # Pattern: @alias with word boundary, case-insensitive, preceded by start/whitespace or punctuation
        pattern = rf"(?i)(?:^|[\s,;:.!?])@{re.escape(alias)}\b"
        if re.search(pattern, text):
            mentioned_alias = alias
            break
        # Also support mention without @ at start like "sahana " at beginning? Spec says @sahana required, so keep @ required
        # But allow bare alias detection for robustness if user types "@Sahana" with different cases already handled

    if not mentioned_alias and not is_reply_to_bot:
        return None

    # Clean all alias mentions from text (remove every @alias occurrence) including trailing punctuation ,:;.!? 
    cleaned_text = text
    for alias in aliases:
        pattern = rf"(?i)\s*@{re.escape(alias)}\b[,\-.:;!?]*\s*"
        cleaned_text = re.sub(pattern, " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    # Strip leading leftover punctuation after removal (e.g., ", what" -> "what")
    cleaned_text = re.sub(r"^[,\-.:;!?]+\s*", "", cleaned_text).strip()

    # If still empty after cleaning, provide a default prompt so bot always replies when mentioned
    if not cleaned_text:
        if message.get("photo") or message.get("document") or message.get("video") or message.get("animation") or message.get("audio") or message.get("sticker"):
            return "Describe this"
        # Bare mention → friendly greeting prompt
        return "Hello"

    return cleaned_text