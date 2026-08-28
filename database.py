import json
import redis.asyncio as redis
from typing import Optional
from config import REDIS_URL, MAX_HISTORY

r = redis.from_url(REDIS_URL, decode_responses=True)

def hk(cid: int) -> str: return f"chat:{cid}:history"
def rsk(cid: int) -> str: return f"chat:{cid}:reply_state"
def sk(cid: int) -> str: return f"chat:{cid}:state"
def fk(cid: int) -> str: return f"chat:{cid}:file"
def mk(cid: int) -> str: return f"chat:{cid}:memories"
def ck(cid: int) -> str: return f"chat:{cid}:agent_context"

async def save_user(uid: int, name: str) -> None: await r.hset("totalUsers", str(uid), name)
async def user_exists(uid: int) -> bool: return await r.hexists("totalUsers", str(uid))

async def remove_all_user_data(uid: int) -> None:
    await r.delete(hk(uid), rsk(uid), sk(uid), fk(uid), mk(uid), ck(uid))
    await r.delete(f"settings:{uid}:system", f"settings:{uid}:voice", f"settings:{uid}:temp", f"settings:{uid}:model", f"settings:{uid}:tools")
    await r.hdel("totalUsers", str(uid))

async def get_all_users() -> dict[str, str]: return await r.hgetall("totalUsers")
async def ban_user(uid: int, name: str) -> None:
    if is_admin(uid):
        return
    await r.hset("bannedUsers", str(uid), name)

async def unban_user(uid: int) -> None: await r.hdel("bannedUsers", str(uid))
async def is_banned(uid: int) -> bool: return await r.hexists("bannedUsers", str(uid))
async def get_banned_users() -> dict[str, str]: return await r.hgetall("bannedUsers")

async def ensure_admin_not_banned() -> None:
    """Repair Redis immediately if any configured admin was banned.

    Admins must never be banned. If stale/corrupt Redis data contains a banned
    admin, clear the entire database so all ban/unban state and related bot data
    is removed before the update continues.
    """
    from config import ADMINS
    banned = await get_banned_users()
    if any(str(admin_id) in banned for admin_id in ADMINS):
        await r.flushdb()

async def clear_full_redis_data() -> None:
    """Clear all data in Redis. Use with caution."""
    await r.flushdb()

async def save_message(cid: int, role: str, text: str) -> None:
    key = hk(cid)
    if await r.llen(key) >= MAX_HISTORY * 2:
        await r.ltrim(key, -((MAX_HISTORY - 1) * 2), -1)
    await r.rpush(key, json.dumps({"role": role, "text": text}))

async def get_all_history(cid: int) -> list[dict]:
    return [json.loads(i) for i in await r.lrange(hk(cid), 0, -1)]

async def get_recent_history(cid: int, count: int) -> list[dict]:
    key = hk(cid)
    total = await r.llen(key)
    if total == 0: return []
    start = max(0, total - count * 2)
    return [json.loads(i) for i in await r.lrange(key, start, -1)]

async def clear_history(cid: int) -> None: await r.delete(hk(cid))
async def set_reply_state(cid: int, target: int) -> None: await r.set(rsk(cid), str(target), ex=3600)
async def get_reply_state(cid: int) -> Optional[int]:
    val = await r.get(rsk(cid))
    return int(val) if val else None
async def clear_reply_state(cid: int) -> None: await r.delete(rsk(cid))
async def set_state(cid: int, st: str) -> None: await r.set(sk(cid), st, ex=3600)
async def get_state(cid: int) -> Optional[str]: return await r.get(sk(cid))
async def clear_state(cid: int) -> None: await r.delete(sk(cid))
async def save_file_data(cid: int, data: dict) -> None: await r.set(fk(cid), json.dumps(data), ex=86400)
async def get_file_data(cid: int) -> Optional[dict]:
    val = await r.get(fk(cid))
    return json.loads(val) if val else None
async def clear_file_data(cid: int) -> None: await r.delete(fk(cid))
async def get_memories(cid: int) -> list[str]:
    try:
        raw = await r.lrange(mk(cid), 0, -1)
        return [str(m).strip() for m in raw if m and str(m).strip()]
    except Exception:
        return []


def format_memories_block(memories: list[str]) -> str:
    """Return memories in a clean formatted numbered list for model/user display."""
    if not memories:
        return "No saved memories found."
    lines = []
    for idx, mem in enumerate(memories, 1):
        cleaned = str(mem).strip()
        if cleaned:
            lines.append(f"{idx}. {cleaned}")
    return "\n".join(lines) if lines else "No saved memories found."


async def get_formatted_memories(cid: int) -> str:
    """Load memories and return a formatted string (numbered list) for display or model context."""
    memories = await get_memories(cid)
    if not memories:
        return "No saved memories found for this user."
    header = f"🧠 Saved Memories ({len(memories)}):\n"
    return header + format_memories_block(memories)


async def save_memory(cid: int, memory: str) -> bool:
    """Save a memory string for cid. Returns True if saved, False if duplicate/empty/failed."""
    cleaned = (memory or "").strip()
    if not cleaned:
        return False
    # Enforce length limit to avoid abuse (max 1000 chars)
    if len(cleaned) > 1000:
        cleaned = cleaned[:1000].strip()
    try:
        memories = await get_memories(cid)
        # Case-insensitive duplicate check
        lowered = cleaned.lower()
        if any(m.lower() == lowered for m in memories):
            return False
        await r.rpush(mk(cid), cleaned)
        current_len = await r.llen(mk(cid))
        if current_len > 50:
            await r.ltrim(mk(cid), current_len - 50, -1)
        return True
    except Exception:
        return False


async def save_memories_batch(cid: int, memories: list[str]) -> dict:
    """Save multiple memories at once, deduplicated case-insensitively.

    Returns dict with counts: {saved: int, duplicates: int, failed: int, total: int, saved_items: list, duplicate_items: list}
    Bulk operation uses pipeline-style dedup and respects 50 limit via trim.
    """
    if not memories or not isinstance(memories, list):
        return {"saved": 0, "duplicates": 0, "failed": 0, "total": 0, "saved_items": [], "duplicate_items": []}
    # Clean and filter
    cleaned_list: list[str] = []
    for m in memories:
        if m is None:
            continue
        cm = str(m).strip()
        if not cm:
            continue
        if len(cm) > 1000:
            cm = cm[:1000].strip()
        if cm:
            cleaned_list.append(cm)
    if not cleaned_list:
        return {"saved": 0, "duplicates": 0, "failed": 0, "total": 0, "saved_items": [], "duplicate_items": []}
    # Also handle case where single string contains newline/semicolon separated? Keep as is — model should send array
    try:
        existing = await get_memories(cid)
        existing_lower = {m.lower() for m in existing}
        # Dedup within incoming list case-insensitively preserving first occurrence
        seen_lower: set[str] = set()
        deduped: list[str] = []
        duplicates: list[str] = []
        for cm in cleaned_list:
            low = cm.lower()
            if low in existing_lower or low in seen_lower:
                duplicates.append(cm)
                continue
            seen_lower.add(low)
            deduped.append(cm)
        saved_items: list[str] = []
        failed = 0
        for cm in deduped:
            try:
                await r.rpush(mk(cid), cm)
                saved_items.append(cm)
            except Exception:
                failed += 1
        # Trim to 50
        try:
            cur_len = await r.llen(mk(cid))
            if cur_len > 50:
                await r.ltrim(mk(cid), cur_len - 50, -1)
        except Exception:
            pass
        return {
            "saved": len(saved_items),
            "duplicates": len(duplicates),
            "failed": failed,
            "total": len(cleaned_list),
            "saved_items": saved_items,
            "duplicate_items": duplicates,
        }
    except Exception:
        return {"saved": 0, "duplicates": 0, "failed": len(cleaned_list), "total": len(cleaned_list), "saved_items": [], "duplicate_items": []}


async def clear_memories(cid: int) -> None: await r.delete(mk(cid))
async def get_user_voice(cid: int) -> str:
    from config import DEFAULT_TTS_VOICE
    return await r.get(f"settings:{cid}:voice") or DEFAULT_TTS_VOICE
async def set_user_voice(cid: int, voice: str) -> None: await r.set(f"settings:{cid}:voice", voice)
async def get_user_system(cid: int) -> str: return await r.get(f"settings:{cid}:system") or ""
async def set_user_system(cid: int, text: str) -> None: await r.set(f"settings:{cid}:system", text)
async def clear_user_system(cid: int) -> None: await r.delete(f"settings:{cid}:system")
async def get_user_temp(cid: int) -> float:
    val = await r.get(f"settings:{cid}:temp")
    return float(val) if val else 0.7
async def set_user_temp(cid: int, temp: float) -> None: await r.set(f"settings:{cid}:temp", str(temp))
async def get_user_model(cid: int) -> str:
    from config import DEFAULT_MODEL
    return await r.get(f"settings:{cid}:model") or DEFAULT_MODEL
async def set_user_model(cid: int, model: str) -> None: await r.set(f"settings:{cid}:model", model)
async def ensure_user(cid: int, name: str) -> None:
    if not await user_exists(cid): await save_user(cid, name)

# User tool preferences — built-in tools removed, only functionDeclarations are used.
# Kept for backward compatibility; returns empty dict.
DEFAULT_USER_TOOLS: dict = {}

async def get_user_tools(cid: int) -> dict:
    # No built-in tools — always return empty; kept for compatibility
    return {}

async def set_user_tools(cid: int, tools: dict) -> None:
    # No-op — built-in tools (googleSearch) removed
    return

def is_admin(uid: int) -> bool:
    from config import ADMINS
    return uid in ADMINS

async def check_banned(cid: int) -> bool: return await is_banned(cid) and not is_admin(cid)
async def get_credit_message() -> str: return await r.get("settings:credit_message") or "Developer: Sahana AI Team\nCredits: Thanks for using Sahana AI."
async def set_credit_message(text: str) -> None: await r.set("settings:credit_message", text)
