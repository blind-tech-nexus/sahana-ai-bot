import re
from typing import Optional
from config import BOT_USERNAME, BOT_MENTION_ALIASES

def is_group_chat(message: dict) -> bool:
    chat_type = (message.get("chat") or {}).get("type", "")
    return chat_type in {"group", "supergroup"}

def extract_group_prompt(message: dict) -> Optional[str]:
    text = (message.get("text") or message.get("caption") or "")
    
    reply_to = message.get("reply_to_message")
    is_reply_to_bot = False
    if reply_to:
        from_user = reply_to.get("from") or {}
        bot_username = (BOT_USERNAME or "").lower().lstrip("@")
        if from_user.get("is_bot") and from_user.get("username", "").lower() == bot_username:
            is_reply_to_bot = True
            
    # Add default aliases including "sahana", "sahanai", "sahanaai"
    aliases = {"ai", "sahana", "sahanai", "sahanaai"}
    for a in BOT_MENTION_ALIASES:
        if a: aliases.add(a.lower().lstrip("@"))
    if BOT_USERNAME:
        aliases.add(BOT_USERNAME.lower().lstrip("@"))
        
    mentioned_alias = None
    for alias in aliases:
        pattern = rf"(?i)(?:^|\s)@{re.escape(alias)}\b"
        if re.search(pattern, text):
            mentioned_alias = alias
            break
            
    if not mentioned_alias and not is_reply_to_bot:
        return None
        
    cleaned_text = text
    if mentioned_alias:
        pattern = rf"(?i)\s*@{re.escape(mentioned_alias)}\b\s*"
        cleaned_text = re.sub(pattern, " ", cleaned_text).strip()
        
    # Only return the prompt if it's a group chat or reply to bot in group
    chat_type = (message.get("chat") or {}).get("type", "")
    if chat_type not in {"group", "supergroup"}:
        return None
        
    return cleaned_text or ("Describe this" if (message.get("photo") or message.get("document")) else None)