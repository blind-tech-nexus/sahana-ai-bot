import os
import random
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

_KEY_ERROR_COOLDOWN = 30.0
_key_error_times: dict[str, float] = {}


async def fetch_api_keys() -> bool:
    global api_keys, LAST_FETCH_TIME
    current_time = time.time()
    if api_keys and (current_time - LAST_FETCH_TIME) < CACHE_TTL:
        return True
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

    env_keys = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEYS") or os.environ.get("GEMINI_API_KEYS")
    if env_keys:
        api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        LAST_FETCH_TIME = current_time
        logger.info(f"Using {len(api_keys)} api keys from environment variables.")
        return bool(api_keys)

    return bool(api_keys)


async def get_next_key_index() -> int:
    global _key_index
    async with _key_lock:
        if not api_keys:
            return 0
        idx = _key_index
        _key_index = (_key_index + 1) % len(api_keys)
        return idx


def is_key_on_cooldown(key: str) -> bool:
    err_time = _key_error_times.get(key)
    if err_time is None:
        return False
    return (time.time() - err_time) < _KEY_ERROR_COOLDOWN


def mark_key_error(key: str) -> None:
    _key_error_times[key] = time.time()


def clear_key_error(key: str) -> None:
    _key_error_times.pop(key, None)


class KeyRotator:
    def __init__(self, start_idx: int, all_keys: list[str]):
        self._keys = all_keys
        self._available = [k for k in all_keys if not is_key_on_cooldown(k)]
        self._start_idx = start_idx
        self._tried = 0

    def get_next_key(self) -> Optional[str]:
        if not self._available or self._tried >= len(self._available):
            return None
        idx = (self._start_idx + self._tried) % len(self._available)
        self._tried += 1
        return self._available[idx]

    @property
    def tried_count(self) -> int:
        return self._tried

    @property
    def available_count(self) -> int:
        return len(self._available)


def is_retriable_error(e: Exception) -> bool:
    err_str = str(e).lower()
    non_retriable = {"400", "401", "403", "404", "invalid", "permission", "denied", "malformed", "bad request", "safety"}
    if any(c in err_str for c in non_retriable):
        return False
    retriable = {"429", "500", "502", "503", "504", "resource_exhausted", "unavailable", "connection", "timeout"}
    if any(c in err_str for c in retriable):
        return True
    return True


def get_retry_after(resp) -> float:
    retry_after = resp.headers.get("Retry-After", "")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    retry_after = resp.headers.get("retry-after", "")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return 0.0


def compute_backoff_delay(attempt: int, max_delay: float = 60.0, jitter: bool = True) -> float:
    delay = min(2 ** attempt, max_delay)
    if jitter:
        delay *= (0.5 + random.random() * 0.5)
    return delay


class TokenBucketRateLimiter:
    """Client-side rate limiter using a token bucket algorithm to prevent 429s."""

    def __init__(self, rate: float = 8.0, capacity: int = 20, per_key: bool = False):
        self._rate = rate
        self._capacity = capacity
        self._per_key = per_key
        self._buckets: dict[str, list] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str = "default", timeout: float = 30.0) -> bool:
        async with self._lock:
            if self._per_key:
                bucket = self._get_or_create_bucket(key)
            else:
                bucket = self._get_or_create_bucket("default")
            deadline = time.monotonic() + timeout
            while True:
                self._refill(bucket)
                if bucket["tokens"] >= 1:
                    bucket["tokens"] -= 1
                    return True
                wait_time = 1.0 / self._rate
                if time.monotonic() + wait_time > deadline:
                    return False
                await asyncio.sleep(wait_time)

    def _get_or_create_bucket(self, key: str) -> dict:
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": self._capacity,
                "last_refill": time.monotonic(),
            }
        return self._buckets[key]

    def _refill(self, bucket: dict) -> None:
        now = time.monotonic()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(self._capacity, bucket["tokens"] + elapsed * self._rate)
        bucket["last_refill"] = now


_rate_limiter: Optional[TokenBucketRateLimiter] = None


def get_rate_limiter() -> TokenBucketRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = TokenBucketRateLimiter(rate=8.0, capacity=20)
    return _rate_limiter
