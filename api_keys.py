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
    global _key_index
    async with _key_lock:
        if not api_keys: return 0
        idx = _key_index
        _key_index = (_key_index + 1) % len(api_keys)
        return idx

class KeyRotator:
    def __init__(self, start_idx: int):
        self._keys = list(api_keys)
        self._start_idx = start_idx
        self._tried = 0

    def get_next_key(self, tried_keys: list[int] | None = None) -> Optional[str]:
        if not self._keys or self._tried >= len(self._keys): return None
        idx = (self._start_idx + self._tried) % len(self._keys)
        self._tried += 1
        # If tried_keys is provided, skip keys that are already in it
        if tried_keys is not None and idx in tried_keys:
            # Try to find next non-tried key
            for offset in range(1, len(self._keys) + 1):
                next_idx = (self._start_idx + self._tried + offset - 1) % len(self._keys)
                if next_idx not in tried_keys:
                    self._tried += offset
                    return self._keys[next_idx]
            return None
        return self._keys[idx]

def is_retriable_error(e: Exception) -> bool:
    err_str = str(e).lower()
    non_retriable = {"400", "401", "403", "404", "invalid", "permission", "denied", "malformed", "bad request", "safety"}
    if any(c in err_str for c in non_retriable): return False
    retriable = {"429", "500", "502", "503", "504", "resource_exhausted", "unavailable", "connection", "timeout"}
    if any(c in err_str for c in retriable): return True
    return True