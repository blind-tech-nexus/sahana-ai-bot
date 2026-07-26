import httpx
import logging
import time
import asyncio
from typing import Optional
from config import POOL_API

logger = logging.getLogger("mero.api_keys")
api_keys: list[str] = []
LAST_FETCH_TIME: float = 0
CACHE_TTL = 300
_key_index = 0
_key_lock = asyncio.Lock()

async def fetch_api_keys() -> bool:
    global api_keys, LAST_FETCH_TIME
    current_time = time.time()
    if api_keys and (current_time - LAST_FETCH_TIME) < CACHE_TTL: return True
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(POOL_API)
            if resp.status_code == 200:
                keys = resp.json()
                if isinstance(keys, list) and keys:
                    api_keys = [k for k in keys if k and isinstance(k, str)]
                    LAST_FETCH_TIME = current_time
                    logger.info(f"Fetched {len(api_keys)} api keys.")
                    return bool(api_keys)
    except Exception as exc:
        logger.warning(f"fetch_api_keys failed: {exc}")
    return bool(api_keys)

async def get_next_key_index() -> int:
    """Return the next round-robin start index for a Gemini request."""
    global _key_index
    async with _key_lock:
        if not api_keys:
            return 0
        idx = _key_index % len(api_keys)
        _key_index = (idx + 1) % len(api_keys)
        return idx

class KeyRotator:
    """Per-request key iterator.

    Each Gemini request starts from the shared round-robin index, then tries
    every currently cached key at most once: key one, key two, etc.  Failed
    indices are recorded in ``tried_keys`` for logging/inspection; skipping is
    intentionally not based on caller state so one bad bookkeeping value cannot
    prematurely exhaust rotation.
    """
    def __init__(self, start_idx: int):
        self._keys = list(api_keys)
        self._start_idx = start_idx % len(self._keys) if self._keys else 0
        self._offset = 0

    def get_next_key(self, tried_keys: list[int] | None = None) -> tuple[Optional[int], Optional[str]]:
        if not self._keys or self._offset >= len(self._keys):
            return None, None
        idx = (self._start_idx + self._offset) % len(self._keys)
        self._offset += 1
        if tried_keys is not None and idx not in tried_keys:
            tried_keys.append(idx)
        return idx, self._keys[idx]

def is_retriable_error(e: Exception) -> bool:
    err_str = str(e).lower()
    non_retriable = {"400", "401", "403", "404", "invalid", "permission", "denied", "malformed", "bad request", "safety"}
    if any(c in err_str for c in non_retriable): return False
    retriable = {"429", "500", "502", "503", "504", "resource_exhausted", "unavailable", "connection", "timeout"}
    if any(c in err_str for c in retriable): return True
    return True
