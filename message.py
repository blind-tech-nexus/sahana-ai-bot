import httpx
import asyncio
from typing import Optional
from config import TELEGRAM_API, BOT_TOKEN

# Global concurrency limiter for all Telegram network operations: max 50 concurrent workers per spec
_TELEGRAM_SEMAPHORE = asyncio.Semaphore(50)
_TELEGRAM_LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=50)


async def send_message(cid: int, text: str, parse_mode: Optional[str] = None, reply_markup: Optional[dict] = None, reply_to_message_id: Optional[int] = None) -> Optional[dict]:
    chunks = [text[i:i + 4096] for i in range(0, len(text), 4096)]
    result = None
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=30.0, limits=_TELEGRAM_LIMITS) as client:
            for idx, chunk in enumerate(chunks):
                payload: dict = {"chat_id": cid, "text": chunk}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                if reply_markup:
                    payload["reply_markup"] = reply_markup
                # Telegram reply support: reply to original group message when Bot is mentioned
                if reply_to_message_id is not None and idx == 0:
                    payload["reply_parameters"] = {"message_id": reply_to_message_id}
                    # Fallback for older API
                    payload["reply_to_message_id"] = reply_to_message_id
                resp = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
                result = resp.json()
                if not result.get("ok") and parse_mode:
                    payload.pop("parse_mode", None)
                    resp = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload)
                    result = resp.json()
    return result


async def send_photo(cid: int, photo_url: str, caption: Optional[str] = None, reply_markup: Optional[dict] = None) -> dict:
    payload: dict = {"chat_id": cid, "photo": photo_url}
    if caption:
        payload["caption"] = caption[:1024]
    if reply_markup:
        payload["reply_markup"] = reply_markup
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=60.0, limits=_TELEGRAM_LIMITS) as client:
            return (await client.post(f"{TELEGRAM_API}/sendPhoto", json=payload)).json()


async def send_voice_bytes(cid: int, audio_bytes: bytes, caption: Optional[str] = None, filename: str = "response.ogg", mime_type: str = "audio/ogg") -> dict:
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=60.0, limits=_TELEGRAM_LIMITS) as client:
            files = {"voice": (filename, audio_bytes, mime_type)}
            data: dict = {"chat_id": str(cid)}
            if caption:
                data["caption"] = caption[:1024]
            return (await client.post(f"{TELEGRAM_API}/sendVoice", files=files, data=data)).json()


async def download_telegram_file(file_id: str) -> Optional[bytes]:
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=60.0, limits=_TELEGRAM_LIMITS) as client:
            info = (await client.get(f"{TELEGRAM_API}/getFile?file_id={file_id}")).json()
            if not info.get("ok"):
                return None
            resp = await client.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{info['result']['file_path']}")
            return resp.content if resp.status_code == 200 else None


async def get_telegram_file_info(file_id: str) -> Optional[dict]:
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=30.0, limits=_TELEGRAM_LIMITS) as client:
            info = (await client.get(f"{TELEGRAM_API}/getFile?file_id={file_id}")).json()
            if info.get("ok"):
                return info["result"]
    return None


async def answer_callback(cb_id: str, text: str = "") -> None:
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=10.0, limits=_TELEGRAM_LIMITS) as client:
            await client.post(f"{TELEGRAM_API}/answerCallbackQuery", json={"callback_query_id": cb_id, "text": text})


async def edit_message(cid: int, mid: int, text: str, parse_mode: Optional[str] = None, reply_markup: Optional[dict] = None) -> None:
    payload: dict = {"chat_id": cid, "message_id": mid, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=10.0, limits=_TELEGRAM_LIMITS) as client:
            resp = await client.post(f"{TELEGRAM_API}/editMessageText", json=payload)
            if not resp.json().get("ok") and parse_mode:
                payload.pop("parse_mode", None)
                await client.post(f"{TELEGRAM_API}/editMessageText", json=payload)


async def delete_message(cid: int, mid: int) -> None:
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=10.0, limits=_TELEGRAM_LIMITS) as client:
            await client.post(f"{TELEGRAM_API}/deleteMessage", json={"chat_id": cid, "message_id": mid})


async def send_chat_action(cid: int, action: str = "typing") -> None:
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=10.0, limits=_TELEGRAM_LIMITS) as client:
            await client.post(f"{TELEGRAM_API}/sendChatAction", json={"chat_id": cid, "action": action})


async def send_document_bytes(cid: int, file_bytes: bytes, filename: str, caption: Optional[str] = None, mime_type: str = "application/octet-stream") -> dict:
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=30.0, limits=_TELEGRAM_LIMITS) as client:
            files = {"document": (filename, file_bytes, mime_type)}
            data: dict = {"chat_id": str(cid)}
            if caption:
                data["caption"] = caption[:1024]
            return (await client.post(f"{TELEGRAM_API}/sendDocument", files=files, data=data)).json()


async def copy_message(to_chat_id: int, from_chat_id: int, message_id: int, reply_markup: Optional[dict] = None) -> Optional[dict]:
    payload: dict = {
        "chat_id": to_chat_id,
        "from_chat_id": from_chat_id,
        "message_id": message_id,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    async with _TELEGRAM_SEMAPHORE:
        async with httpx.AsyncClient(timeout=30.0, limits=_TELEGRAM_LIMITS) as client:
            resp = await client.post(f"{TELEGRAM_API}/copyMessage", json=payload)
            return resp.json()


# Helper to send many Telegram messages concurrently with max 50 workers
async def send_messages_concurrent(tasks: list) -> list:
    """Execute a list of async Telegram send coroutines concurrently with max 50 workers."""
    if not tasks:
        return []
    sem = asyncio.Semaphore(50)

    async def _run(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[_run(t) for t in tasks], return_exceptions=False)
