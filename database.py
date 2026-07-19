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
    await r.delete(f"settings:{uid}:system", f"settings:{uid}:voice", f"settings:{uid}:temp", f"settings:{uid}:model")
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
    """Check all admins and unban them if banned, then clear all ban-related data."""
    from config import ADMINS
    banned = await get_banned_users()
    for admin_id in ADMINS:
        if str(admin_id) in banned:
            await unban_user(admin_id)
    if banned:
        await r.delete("bannedUsers")

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
async def get_memories(cid: int) -> list[str]: return [m for m in await r.lrange(mk(cid), 0, -1) if m]

async def save_memory(cid: int, memory: str) -> None:
    cleaned = (memory or "").strip()
    if not cleaned: return
    memories = await get_memories(cid)
    if cleaned in memories: return
    await r.rpush(mk(cid), cleaned)
    if await r.llen(mk(cid)) > 50: await r.ltrim(mk(cid), 1, -1)

async def clear_memories(cid: int) -> None: await r.delete(mk(cid))
async def get_user_voice(cid: int) -> str: return await r.get(f"settings:{cid}:voice") or "en"
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

def is_admin(uid: int) -> bool:
    from config import ADMINS
    return uid in ADMINS

async def check_banned(cid: int) -> bool: return await is_banned(cid) and not is_admin(cid)
async def get_credit_message() -> str: return await r.get("settings:credit_message") or "Developer: Sahana AI Team\nCredits: Thanks for using Sahana AI."
async def set_credit_message(text: str) -> None: await r.set("settings:credit_message", text)