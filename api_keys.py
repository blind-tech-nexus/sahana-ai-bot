import os
import random
import httpx
import logging
import time
import asyncio
import threading
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from config import POOL_API

logger = logging.getLogger("mero.api_keys")

api_keys: list[str] = []
LAST_FETCH_TIME: float = 0
CACHE_TTL = 300

# Cooldown map: key -> expiry timestamp (time.time())
_DEFAULT_COOLDOWN_429 = 60.0
_DEFAULT_COOLDOWN_5XX = 15.0
_DEFAULT_COOLDOWN_AUTH = 300.0
_DEFAULT_COOLDOWN_GENERIC = 30.0

_key_cooldown: dict[str, float] = {}
_key_cooldown_lock = threading.Lock()

# Backward compat aliases (old names)
_KEY_ERROR_COOLDOWN = _DEFAULT_COOLDOWN_GENERIC
_key_error_times = _key_cooldown
_key_error_lock = _key_cooldown_lock

# Async lock for fetching/rotating keys
_fetch_lock = asyncio.Lock()

# Unified round-robin index (single source for both async and sync)
_global_key_index = 0
_global_key_lock = threading.Lock()

# Global concurrency limiters for all networking: max 50 workers per spec
_API_SEMAPHORE = asyncio.Semaphore(50)
_API_LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=50)


async def fetch_api_keys() -> bool:
    """Fetch API keys from the pool endpoint or environment, caching with TTL."""
    global api_keys, LAST_FETCH_TIME
    async with _fetch_lock:
        current_time = time.time()
        if api_keys and (current_time - LAST_FETCH_TIME) < CACHE_TTL:
            return True
        try:
            async with _API_SEMAPHORE:
                async with httpx.AsyncClient(timeout=30.0, limits=_API_LIMITS) as client:
                    resp = await client.get(POOL_API)
                if resp.status_code == 200:
                    keys = None
                    # Try JSON first
                    try:
                        data = resp.json()
                    except Exception:
                        data = None
                    if isinstance(data, list):
                        keys = data
                    elif isinstance(data, dict):
                        # Support various pool API response shapes
                        for k in ("keys", "api_keys", "data", "result", "apiKeys"):
                            v = data.get(k)
                            if isinstance(v, list) and v:
                                keys = v
                                break
                        if keys is None:
                            # try values that look like keys
                            vals = [v for v in data.values() if isinstance(v, str) and v.startswith("AIza")]
                            if vals:
                                keys = vals
                    # Fallback: raw text split (newline/comma/semicolon)
                    if keys is None:
                        try:
                            raw_text = resp.text or ""
                            if raw_text.strip():
                                # If text looks like JSON string? try split
                                # Split by common delimiters and look for AIza
                                parts = re.split(r"[,\n;]+", raw_text)
                                candidates = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
                                # Filter plausible keys: startswith AIza or length >30
                                filtered_txt = [c for c in candidates if c.startswith("AIza") or (len(c) > 30 and " " not in c)]
                                if filtered_txt:
                                    keys = filtered_txt
                        except Exception:
                            pass
                    if isinstance(keys, list) and keys:
                        filtered = [str(k).strip() for k in keys if k and isinstance(k, (str,))]
                        filtered = [k.strip().strip('"').strip("'") for k in filtered if k.strip()]
                        # Remove empty and duplicates preserving order
                        seen = set()
                        uniq = []
                        for k in filtered:
                            if k not in seen:
                                seen.add(k)
                                uniq.append(k)
                        filtered = uniq
                        if filtered:
                            # Clean cooldown entries for keys no longer in pool
                            with _key_cooldown_lock:
                                for old_k in list(_key_cooldown.keys()):
                                    if old_k not in filtered:
                                        _key_cooldown.pop(old_k, None)
                            api_keys = filtered
                            LAST_FETCH_TIME = current_time
                            logger.info(f"Fetched {len(api_keys)} api keys from pool.")
                            return True
                    logger.warning(f"Pool API returned unexpected shape: {str(data)[:300] if data is not None else resp.text[:300]}")
        except Exception as exc:
            logger.warning(f"fetch_api_keys from {POOL_API} failed: {exc}")

        env_keys_str = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEYS") or os.environ.get("GEMINI_API_KEYS")
        if env_keys_str:
            # Support comma, semicolon, newline separated
            parts = re.split(r"[,\n;]+", env_keys_str)
            keys = [k.strip() for k in parts if k.strip()]
            if keys:
                api_keys = keys
                LAST_FETCH_TIME = current_time
                logger.info(f"Using {len(api_keys)} api keys from environment variables.")
                return bool(api_keys)

        if not api_keys:
            logger.error("No API keys available after fetch.")
        return bool(api_keys)


def _now_ts() -> float:
    return time.time()


def is_key_on_cooldown(key: str) -> bool:
    """Check if a key is on cooldown (thread-safe)."""
    with _key_cooldown_lock:
        expiry = _key_cooldown.get(key)
        if expiry is None:
            return False
        if _now_ts() >= expiry:
            # expired, clean up
            _key_cooldown.pop(key, None)
            return False
        return True


def mark_key_error(key: str, cooldown_seconds: Optional[float] = None) -> None:
    """Mark a key as errored and put it on cooldown for given duration."""
    if cooldown_seconds is None:
        cooldown_seconds = _DEFAULT_COOLDOWN_GENERIC
    # Clamp to reasonable bounds
    cooldown_seconds = max(1.0, min(float(cooldown_seconds), 600.0))
    expiry = _now_ts() + cooldown_seconds
    with _key_cooldown_lock:
        # Keep longest expiry if already cooling down
        existing = _key_cooldown.get(key)
        if existing is not None and existing > expiry:
            return
        _key_cooldown[key] = expiry
    logger.debug(f"Key ...{key[-6:]} on cooldown for {cooldown_seconds:.1f}s until {expiry:.0f}")


def clear_key_error(key: str) -> None:
    """Clear the error cooldown state for a key."""
    with _key_cooldown_lock:
        _key_cooldown.pop(key, None)


def get_available_keys() -> list[str]:
    """Return list of keys not currently on cooldown (thread-safe snapshot)."""
    now = _now_ts()
    with _key_cooldown_lock:
        # clean expired entries
        expired = [k for k, exp in _key_cooldown.items() if now >= exp]
        for k in expired:
            _key_cooldown.pop(k, None)
        cooldown_snapshot = dict(_key_cooldown)
    return [k for k in api_keys if cooldown_snapshot.get(k, 0) <= now]


def get_cooldown_stats() -> dict:
    """Return diagnostic info about pool health (thread-safe)."""
    now = _now_ts()
    with _key_cooldown_lock:
        expiries = {k: max(0.0, exp - now) for k, exp in _key_cooldown.items() if exp > now}
    total = len(api_keys)
    available = len(get_available_keys())
    return {
        "total": total,
        "available": available,
        "cooldown_count": len(expiries),
        "cooldown_remaining": expiries,
    }


def time_until_next_key_available() -> float:
    """Seconds until the next key exits cooldown, 0 if any available now."""
    if get_available_keys():
        return 0.0
    now = _now_ts()
    with _key_cooldown_lock:
        expiries = [exp - now for exp in _key_cooldown.values() if exp > now]
        if not expiries:
            return 0.0
        return min(expiries)


def next_available_in() -> float:
    """Alias for time_until_next_key_available for backward compat."""
    return time_until_next_key_available()


def _next_index() -> int:
    """Thread-safe global round-robin index increment."""
    global _global_key_index
    with _global_key_lock:
        if not api_keys:
            idx = _global_key_index
            _global_key_index += 1
            return idx
        idx = _global_key_index % len(api_keys)
        _global_key_index = (_global_key_index + 1) % len(api_keys) if len(api_keys) > 0 else _global_key_index + 1
        return idx


# Backward compatibility aliases
_key_index = 0
_key_lock = asyncio.Lock()
_key_index_sync = 0
_key_index_sync_lock = threading.Lock()


async def get_next_key_index() -> int:
    """Async round-robin: returns index of next key, advancing the shared counter."""
    # Use unified counter
    return _next_index()


def _get_next_sync_key_index() -> int:
    """Sync round-robin key index for use in synchronous worker threads."""
    return _next_index()


class KeyRotator:
    """Deterministic round-robin key rotator that skips keys on cooldown.

    The rotator is instantiated per-request with a start index and a snapshot
    of the key pool. Each call to get_next_key() returns the next available
    (not on cooldown and not already tried in this rotation cycle) key,
    advancing the internal pointer. Once all keys have been tried, returns None.

    This ensures every key is tried at most once per request, and cooldown
    keys are skipped transparently without wasting retry attempts.
    """

    def __init__(self, start_idx: int = 0, keys_snapshot: Optional[list[str]] = None):
        # snapshot at creation time to keep rotation consistent even if global pool refreshes
        if keys_snapshot is not None:
            self._keys = list(keys_snapshot)
        else:
            self._keys = list(api_keys)
        n = len(self._keys)
        self._pos = (start_idx % n) if n else 0
        self._attempted: set[str] = set()
        self._tried = 0

    def _available_snapshot(self) -> list[str]:
        return get_available_keys()

    def get_next_key(self) -> Optional[str]:
        """Get the next available key, skipping keys on cooldown and already tried."""
        if not self._keys:
            return None
        n = len(self._keys)
        # First pass: try to find not-on-cooldown and not-yet-attempted
        for _ in range(n):
            key = self._keys[self._pos % n]
            self._pos = (self._pos + 1) % n
            if key in self._attempted:
                continue
            if is_key_on_cooldown(key):
                continue
            self._attempted.add(key)
            self._tried += 1
            return key
        # Second pass fallback: if all remaining are on cooldown but we still have untried keys,
        # we could return cooldown keys as last resort if we want to exhaust pool.
        # However, to avoid hammering quota-exhausted keys, we return None when all available are tried.
        # The caller will then decide to wait for cooldown or fail.
        # For completeness, if we have untried keys that are on cooldown and total tried < n,
        # we still allow one more pass that includes cooldown keys (so caller can attempt all keys once)
        # Uncomment below if you want to also try cooldown keys after exhausting healthy ones:
        # for _ in range(n):
        #     key = self._keys[self._pos % n]
        #     self._pos = (self._pos + 1) % n
        #     if key in self._attempted:
        #         continue
        #     self._attempted.add(key)
        #     self._tried += 1
        #     return key
        return None

    def get_next_key_allow_cooldown(self) -> Optional[str]:
        """Variant that returns next untried key even if on cooldown (used for upload fallback)."""
        if not self._keys:
            return None
        n = len(self._keys)
        for _ in range(n):
            key = self._keys[self._pos % n]
            self._pos = (self._pos + 1) % n
            if key in self._attempted:
                continue
            self._attempted.add(key)
            self._tried += 1
            return key
        return None

    @property
    def tried_count(self) -> int:
        return self._tried

    @property
    def available_count(self) -> int:
        # counts keys in snapshot that are not on cooldown and not yet attempted
        return len([k for k in self._keys if k not in self._attempted and not is_key_on_cooldown(k)])

    @property
    def total_count(self) -> int:
        return len(self._keys)

    @property
    def has_remaining(self) -> bool:
        return len(self._attempted) < len(self._keys)


def is_retriable_error(e: Exception) -> bool:
    """Classify whether exception/error is retriable with next key."""
    err_str = str(e).lower()
    # Non-retriable: client errors that won't succeed with different key (except auth which is per-key)
    # 400, 404, invalid etc. are not retryable across keys.
    # 401/403 ARE retriable by rotating to next key (different key may be valid)
    # So only 400/404 are truly non-retriable.
    non_retriable = {"400", "404", "invalid", "permission", "denied", "malformed", "bad request", "safety", "blocked"}
    # Check retriable first
    retriable = {"429", "500", "502", "503", "504", "408", "resource_exhausted", "unavailable", "overloaded", "connection", "timeout", "deadline", "reset", "429"}
    if any(code in err_str for code in retriable):
        return True
    if any(code in err_str for code in non_retriable):
        # 401/403 are not in non_retriable here; they are handled specially as key-specific
        if "401" in err_str or "403" in err_str:
            return True  # rotate to next key
        return False
    # default to retriable for unknown transient-like?
    return True


def _is_auth_error(status_code: int, error_text: str) -> bool:
    return status_code in (401, 403) or "401" in error_text or "403" in error_text or "permission_denied" in error_text.lower() or "api_key_invalid" in error_text.lower()


class HTTPException(Exception):
    """Wraps HTTP errors with status code and error payload for inspection."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


def get_retry_after(resp, error_text: Optional[str] = None) -> float:
    """Parse Retry-After delay from headers and JSON body.

    Checks:
    - Retry-After header (seconds)
    - retry-after lower header
    - JSON body field retryDelay (e.g., "30s") or retry_delay
    - X-Retry-After
    """
    headers = getattr(resp, "headers", {}) or {}
    # headers may be case-insensitive; try common variants
    for hdr in ("Retry-After", "retry-after", "Retry-after", "X-Retry-After", "x-retry-after"):
        val = headers.get(hdr) if isinstance(headers, dict) else None
        if val is None and hasattr(headers, "get"):
            try:
                val = headers.get(hdr)
            except Exception:
                val = None
        if val:
            try:
                # Could be http-date or seconds; we handle seconds only
                return float(str(val).strip())
            except ValueError:
                pass
    # Try body parsing
    text = error_text
    if text is None:
        try:
            text = resp.text if hasattr(resp, "text") else ""
        except Exception:
            text = ""
    if not text:
        return 0.0
    # Search for retryDelay pattern "retryDelay":"39s" or "retryDelay": "39s"
    try:
        # Fast regex search
        m = re.search(r'"retryDelay"\s*:\s*"([^"]+)"', text)
        if m:
            delay_str = m.group(1).strip()
            # e.g., "39s", "60s", "1.5s"
            if delay_str.endswith("s"):
                try:
                    return float(delay_str[:-1])
                except ValueError:
                    pass
            else:
                try:
                    return float(delay_str)
                except ValueError:
                    pass
        m2 = re.search(r'"retry_delay"\s*:\s*"?([0-9.]+)s?"?', text, re.IGNORECASE)
        if m2:
            try:
                return float(m2.group(1))
            except ValueError:
                pass
        # Also try to json parse and walk recursively
        try:
            data = json.loads(text)
            # Walk recursively for retryDelay
            def _walk(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() in ("retrydelay", "retry_delay", "retryafter"):
                            if isinstance(v, str) and v.strip():
                                s = v.strip()
                                if s.endswith("s"):
                                    try:
                                        return float(s[:-1])
                                    except:
                                        pass
                                try:
                                    return float(s)
                                except:
                                    pass
                            elif isinstance(v, (int, float)):
                                return float(v)
                        else:
                            res = _walk(v)
                            if res is not None:
                                return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = _walk(item)
                        if res is not None:
                            return res
                return None
            found = _walk(data)
            if found is not None:
                return float(found)
        except Exception:
            pass
    except Exception:
        pass
    return 0.0


def compute_backoff_delay(attempt: int, max_delay: float = 60.0, jitter: bool = True) -> float:
    """Exponential backoff with jitter."""
    delay = min(2 ** attempt, max_delay)
    if jitter:
        # full jitter: random between 0.5*delay and delay
        delay *= (0.5 + random.random() * 0.5)
    return delay


class TokenBucketRateLimiter:
    """Client-side rate limiter using a token bucket algorithm to prevent 429s.

    Per-key limiting is enabled by default, so each API key has its own bucket.
    This fixes the prior global-bucket bottleneck (8 req/s global) that under-utilized
    multiple keys, and also prevents holding the lock during sleep.
    """

    def __init__(self, rate: float = 2.0, capacity: int = 12, per_key: bool = True):
        # rate: tokens per second per bucket. 2/sec ~=120 RPM per key, capacity 12 burst
        # For Gemini free tier ~60 RPM, use ~1 per sec with burst. We choose 1.5-2 to stay safe but not too throttle.
        # If per_key=True, each key gets its own bucket.
        self._rate = rate
        self._capacity = capacity
        self._per_key = per_key
        self._buckets: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str = "default", timeout: float = 5.0) -> bool:
        # Ensure bucket exists (brief lock)
        async with self._lock:
            bucket_key = key if self._per_key else "default"
            bucket = self._buckets.get(bucket_key)
            if bucket is None:
                bucket = {
                    "tokens": float(self._capacity),
                    "last_refill": time.monotonic(),
                }
                self._buckets[bucket_key] = bucket

        deadline = time.monotonic() + float(timeout)
        while True:
            # Check/refill under lock, but don't sleep while holding lock
            async with self._lock:
                self._refill(bucket)
                if bucket["tokens"] >= 1.0:
                    bucket["tokens"] -= 1.0
                    return True
                # not enough tokens, estimate wait needed
                needed = 1.0 - bucket["tokens"]
                # time to generate needed tokens
                wait_time = needed / self._rate if self._rate > 0 else 1.0
                # Cap wait to avoid long sleep loops
                wait_time = min(max(wait_time, 0.05), 1.0)

            if time.monotonic() + wait_time > deadline:
                return False
            await asyncio.sleep(wait_time)

    def _get_or_create_bucket(self, key: str) -> dict:
        # legacy sync helper (not used in async acquire)
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": float(self._capacity),
                "last_refill": time.monotonic(),
            }
        return self._buckets[key]

    def _refill(self, bucket: dict) -> None:
        now = time.monotonic()
        elapsed = now - bucket["last_refill"]
        if elapsed <= 0:
            return
        bucket["tokens"] = min(float(self._capacity), bucket["tokens"] + elapsed * self._rate)
        bucket["last_refill"] = now


_rate_limiter: Optional[TokenBucketRateLimiter] = None


def get_rate_limiter() -> TokenBucketRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        # Per-key limiter: 1.0 req/s ~60 RPM, burst 8. Conservative to stay under free-tier quota.
        # Per-key ensures one bucket per API key, so total throughput scales with number of keys.
        # Env overrides allowed: GEMINI_RATE_LIMIT_RPS and GEMINI_RATE_LIMIT_BURST
        try:
            rate = float(os.environ.get("GEMINI_RATE_LIMIT_RPS", "1.0"))
            burst = int(os.environ.get("GEMINI_RATE_LIMIT_BURST", "8"))
        except Exception:
            rate, burst = 1.0, 8
        _rate_limiter = TokenBucketRateLimiter(rate=rate, capacity=burst, per_key=True)
        logger.info(f"Rate limiter init: {rate} rps per key, burst {burst}")
    return _rate_limiter


# --- genai.Client pool + concurrent upload helpers ---

_genai_clients: dict[str, "object"] = {}
_genai_clients_lock = threading.Lock()


def get_genai_client(api_key: str):
    """Get or create a genai.Client for a given API key (thread-safe, singleton per key)."""
    with _genai_clients_lock:
        client = _genai_clients.get(api_key)
        if client is not None:
            return client
    from google import genai
    try:
        client = genai.Client(api_key=api_key, http_options={"max_retries": 3})
    except TypeError:
        client = genai.Client(api_key=api_key)
    with _genai_clients_lock:
        existing = _genai_clients.get(api_key)
        if existing is not None:
            return existing
        _genai_clients[api_key] = client
    return client


def get_next_genai_client() -> Optional["object"]:
    """Get the next available genai.Client using key rotation (thread-safe)."""
    if not api_keys:
        return None
    rotator = KeyRotator(_next_index(), list(api_keys))
    key = rotator.get_next_key()
    # fallback to allow cooldown keys if no healthy
    if key is None:
        key = rotator.get_next_key_allow_cooldown()
    if key is None:
        return None
    return get_genai_client(key)


async def _gemini_request(
    url: str,
    payload: dict,
    start_idx: int,
) -> tuple[Optional[dict], Optional[HTTPException]]:
    """Make a Gemini REST API request, rotating keys on retriable errors.

    Correct rotation logic:
    - Tries each distinct API key at most once per call (up to len(api_keys) attempts)
    - Skips keys on cooldown transparently
    - Parses Retry-After / retryDelay and applies per-key cooldown accordingly
    - Uses per-key client-side rate limiting without blocking other keys
    - Small jitter between rotations to avoid thundering herd; does NOT sleep full backoff
      between keys (only when all keys exhausted)
    - On success, clears cooldown for that key
    Returns (parsed_json, error). Parsed JSON is the raw API response dict.
    """
    if not api_keys:
        # Try to fetch once lazily
        fetched = await fetch_api_keys()
        if not fetched or not api_keys:
            return None, HTTPException(0, "No API keys available")

    keys_snapshot = list(api_keys)
    if not keys_snapshot:
        return None, HTTPException(0, "No API keys available")

    rotator = KeyRotator(start_idx, keys_snapshot)
    limiter = get_rate_limiter()
    # Try each distinct key once (no artificial cap at 6)
    max_attempts = len(keys_snapshot)
    last_error: Optional[HTTPException] = None

    # Concurrency per spec: all networking inside max 50 workers
    async with _API_SEMAPHORE:
        async with httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=50),
        ) as client:
            for attempt in range(max_attempts):
                key = rotator.get_next_key()
                if key is None:
                    # All healthy keys tried; optionally try cooldown keys as last resort
                    # If we have not yet tried all keys (including cooldown), try them
                    if rotator.has_remaining:
                        key = rotator.get_next_key_allow_cooldown()
                        if key is None:
                            break
                    else:
                        break
                # Per-key rate limiting: short timeout, if throttled locally skip to next key
                if not await limiter.acquire(key, timeout=2.5):
                    logger.warning(f"Per-key rate limiter throttled key ...{key[-4:]}, rotating")
                    # Light cooldown so we don't spin immediately
                    mark_key_error(key, cooldown_seconds=2.0)
                    last_error = HTTPException(429, "Client-side rate limit (per-key bucket empty)")
                    # small jitter before next key
                    await asyncio.sleep(random.uniform(0.05, 0.2))
                    continue

                full_url = f"{url}?key={key}"
                try:
                    resp = await client.post(full_url, json=payload, headers={"Content-Type": "application/json"})
                    if resp.status_code == 200:
                        clear_key_error(key)
                        try:
                            data = resp.json()
                            return data, None
                        except Exception as exc:
                            return None, HTTPException(resp.status_code, f"Invalid JSON response: {resp.text[:500]} ({exc})")

                    error_text = resp.text
                    logger.warning(
                        f"Gemini API failed key ...{key[-4:]} (attempt {attempt+1}/{max_attempts}) status {resp.status_code}: {error_text[:400]}"
                    )
                    last_error = HTTPException(resp.status_code, error_text)

                    if resp.status_code == 429:
                        retry_after = get_retry_after(resp, error_text)
                        cooldown = retry_after if retry_after > 0 else _DEFAULT_COOLDOWN_429
                        # Also handle quota vs per-minute: if retryDelay > 30s, use that
                        mark_key_error(key, cooldown_seconds=cooldown)
                        logger.info(f"429 on key ...{key[-4:]} -> cooldown {cooldown:.1f}s, rotating ({attempt+1}/{max_attempts})")
                        if attempt < max_attempts - 1:
                            # Brief pause before next key to avoid hammering
                            await asyncio.sleep(random.uniform(0.15, 0.45))
                            continue
                        # All keys exhausted with 429: optionally wait for earliest cooldown if short
                        # Check if any key recovers within ~5s, wait; otherwise fail fast
                        # Find earliest expiry
                        with _key_cooldown_lock:
                            now = _now_ts()
                            expiries = [exp for exp in _key_cooldown.values() if exp > now]
                        if expiries:
                            wait_for = min(expiries) - now
                            if 0 < wait_for <= 5.0 and attempt == max_attempts -1:
                                logger.info(f"All keys on cooldown, waiting {wait_for:.1f}s for next available")
                                await asyncio.sleep(wait_for + 0.2)
                                # try one more time with fresh rotator? Instead break and caller will see error
                        break

                    if resp.status_code in (500, 502, 503, 504, 408):
                        mark_key_error(key, cooldown_seconds=_DEFAULT_COOLDOWN_5XX)
                        if attempt < max_attempts - 1:
                            # small backoff before rotating, not full exponential to keep rotation fast
                            delay = compute_backoff_delay(attempt, max_delay=6.0, jitter=True)
                            # cap per-rotation delay to 1.5s so we can try next key quickly
                            await asyncio.sleep(min(delay, 1.2))
                            continue
                        break

                    if resp.status_code in (401, 403):
                        # Key-specific auth failure: disable for longer and rotate
                        mark_key_error(key, cooldown_seconds=_DEFAULT_COOLDOWN_AUTH)
                        logger.warning(f"Auth error {resp.status_code} on key ...{key[-4:]}, disabled 5m, rotating")
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(random.uniform(0.1, 0.3))
                            continue
                        break

                    # 400, 404 etc - non-retriable across keys, don't rotate further
                    if resp.status_code in (400, 404):
                        logger.warning(f"Non-retriable {resp.status_code}, not rotating further")
                        break

                    # Other 4xx - break
                    if 400 <= resp.status_code < 500:
                        break

                    # Unknown: treat as retryable once?
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                        continue
                    break

                except httpx.TimeoutException:
                    last_error = HTTPException(0, "Request timed out")
                    logger.warning(f"Timeout on API call attempt {attempt+1} key ...{key[-4:]}")
                    mark_key_error(key, cooldown_seconds=10.0)
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(min(compute_backoff_delay(attempt, max_delay=8.0, jitter=True), 1.0))
                        continue
                    break
                except Exception as exc:
                    last_error = HTTPException(0, str(exc))
                    logger.warning(f"API call exception attempt {attempt+1} key ...{key[-4:]}: {exc}")
                    # classify if retriable
                    if is_retriable_error(exc):
                        mark_key_error(key, cooldown_seconds=10.0)
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(min(compute_backoff_delay(attempt, max_delay=8.0, jitter=True), 1.0))
                            continue
                    break

    return None, last_error or HTTPException(0, "All keys exhausted")


def _sync_gemini_request(
    url: str,
    payload: dict,
    start_idx: int,
) -> tuple[Optional[dict], Optional[HTTPException]]:
    """Synchronous version of _gemini_request for use in ThreadPoolExecutor workers."""
    if not api_keys:
        # No async fetch here; assume keys already fetched. If empty, fail.
        return None, HTTPException(0, "No API keys available")

    keys_snapshot = list(api_keys)
    if not keys_snapshot:
        return None, HTTPException(0, "No API keys available")

    rotator = KeyRotator(start_idx, keys_snapshot)
    max_attempts = len(keys_snapshot)
    last_error: Optional[HTTPException] = None

    with httpx.Client(timeout=120.0, limits=httpx.Limits(max_connections=50, max_keepalive_connections=50)) as client:
        for attempt in range(max_attempts):
            key = rotator.get_next_key()
            if key is None:
                if rotator.has_remaining:
                    key = rotator.get_next_key_allow_cooldown()
                    if key is None:
                        break
                else:
                    break

            full_url = f"{url}?key={key}"
            try:
                resp = client.post(full_url, json=payload, headers={"Content-Type": "application/json"})
                if resp.status_code == 200:
                    clear_key_error(key)
                    try:
                        data = resp.json()
                        return data, None
                    except Exception as exc:
                        return None, HTTPException(resp.status_code, f"Invalid JSON response: {resp.text[:500]} ({exc})")

                error_text = resp.text
                logger.warning(
                    f"Gemini API sync failed key ...{key[-4:]} attempt {attempt+1}/{max_attempts} status {resp.status_code}: {error_text[:400]}"
                )
                last_error = HTTPException(resp.status_code, error_text)

                if resp.status_code == 429:
                    retry_after = get_retry_after(resp, error_text)
                    cooldown = retry_after if retry_after > 0 else _DEFAULT_COOLDOWN_429
                    mark_key_error(key, cooldown_seconds=cooldown)
                    if attempt < max_attempts - 1:
                        time.sleep(random.uniform(0.15, 0.45))
                        continue
                    break

                if resp.status_code in (500, 502, 503, 504, 408):
                    mark_key_error(key, cooldown_seconds=_DEFAULT_COOLDOWN_5XX)
                    if attempt < max_attempts - 1:
                        time.sleep(min(compute_backoff_delay(attempt, max_delay=6.0, jitter=True), 1.2))
                        continue
                    break

                if resp.status_code in (401, 403):
                    mark_key_error(key, cooldown_seconds=_DEFAULT_COOLDOWN_AUTH)
                    if attempt < max_attempts - 1:
                        time.sleep(random.uniform(0.1, 0.3))
                        continue
                    break

                if resp.status_code in (400, 404):
                    break
                if 400 <= resp.status_code < 500:
                    break
                if attempt < max_attempts - 1:
                    time.sleep(random.uniform(0.2, 0.5))
                    continue
                break

            except httpx.TimeoutException:
                last_error = HTTPException(0, "Request timed out")
                logger.warning(f"Timeout on sync API call attempt {attempt+1}")
                mark_key_error(key, cooldown_seconds=10.0)
                if attempt < max_attempts - 1:
                    time.sleep(min(compute_backoff_delay(attempt, max_delay=8.0, jitter=True), 1.0))
                    continue
                break
            except Exception as exc:
                last_error = HTTPException(0, str(exc))
                logger.warning(f"Sync API call exception attempt {attempt+1}: {exc}")
                if is_retriable_error(exc) and attempt < max_attempts - 1:
                    mark_key_error(key, cooldown_seconds=10.0)
                    time.sleep(min(compute_backoff_delay(attempt, max_delay=8.0, jitter=True), 1.0))
                    continue
                break

    return None, last_error or HTTPException(0, "All keys exhausted")


def upload_file_concurrency(
    file_contents: bytes,
    mime_type: str,
    display_name: str = "file",
    max_workers: int = 50,
) -> Optional[str]:
    """Upload a single file to Gemini Files API with proper key rotation.

    Previous implementation launched up to 50 concurrent uploads of the SAME file
    against different keys simultaneously, hammering quotas and causing 429s.
    New implementation does sequential rotation trying each healthy key once,
    with proper cooldown handling, up to min(len(keys), max_workers) attempts.
    Concurrency limit: max_workers=50 per spec — all networking inside concurrency with 50 workers.

    Returns file URI (e.g., 'files/abc-123') on success, None on failure.
    """
    if not api_keys:
        return None

    mime_type = normalize_mime_type(mime_type)
    # Limit attempts to min(pool size, max_workers)
    keys_snapshot = list(api_keys)
    max_attempts = min(len(keys_snapshot), max_workers) if max_workers else len(keys_snapshot)
    # Use rotator starting at next index
    rotator = KeyRotator(_next_index(), keys_snapshot)
    last_err = None

    for attempt in range(max_attempts):
        key = rotator.get_next_key()
        if key is None:
            # If healthy exhausted but we still have cooldown keys, try them as fallback
            # Only if we haven't tried all keys
            if rotator.has_remaining:
                key = rotator.get_next_key_allow_cooldown()
            if key is None:
                break
        client = get_genai_client(key)
        try:
            # genai upload is blocking sync
            file_obj = client.files.upload(
                file={"file": ("upload.bin", file_contents), "mime_type": mime_type},
                config={"display_name": display_name[:100]},
            )
            name = getattr(file_obj, "name", None)
            if name:
                # normalize prefix
                if not name.startswith("files/"):
                    name = f"files/{name}"
                clear_key_error(key)
                return name
            # No name? treat as failure and rotate
            logger.warning(f"Upload returned no name for key ...{key[-4:]}: {file_obj}")
            mark_key_error(key, cooldown_seconds=10.0)
        except Exception as exc:
            err_str = str(exc).lower()
            logger.warning(f"File upload failed key ...{key[-4:]} attempt {attempt+1}/{max_attempts}: {exc}")
            last_err = exc
            # Classify error
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                # Try to parse retry delay from message
                retry_after = 0.0
                m = re.search(r'retryDelay.*?([0-9.]+)s', err_str)
                if m:
                    try:
                        retry_after = float(m.group(1))
                    except:
                        retry_after = 0.0
                cooldown = retry_after if retry_after > 0 else _DEFAULT_COOLDOWN_429
                mark_key_error(key, cooldown_seconds=cooldown)
            elif "401" in err_str or "403" in err_str or "invalid" in err_str or "permission" in err_str:
                mark_key_error(key, cooldown_seconds=_DEFAULT_COOLDOWN_AUTH)
            elif "500" in err_str or "502" in err_str or "503" in err_str or "504" in err_str or "unavailable" in err_str:
                mark_key_error(key, cooldown_seconds=_DEFAULT_COOLDOWN_5XX)
            else:
                # generic; light cooldown to avoid immediate reuse
                mark_key_error(key, cooldown_seconds=10.0)
            # brief jitter before next attempt
            time.sleep(random.uniform(0.15, 0.4))
            continue

    if last_err:
        logger.warning(f"All upload attempts failed ({max_attempts} keys tried)")
    return None


async def upload_file_to_gemini(
    file_bytes: bytes,
    mime_type: str,
    display_name: str = "file",
) -> Optional[str]:
    """Upload a file to the Gemini Files API and return the file URI.

    Uses the Files API for files larger than MAX_INLINE_FILE_BYTES (2MB).
    Returns the file URI (e.g. 'files/abc-123') or None on failure.
    """
    from api import MAX_INLINE_FILE_BYTES

    if not await fetch_api_keys():
        return None

    mime_type = normalize_mime_type(mime_type)
    file_size = len(file_bytes)

    if file_size <= MAX_INLINE_FILE_BYTES:
        return None

    loop = asyncio.get_event_loop()
    # Run the synchronous upload helper in a thread pool so we don't block the event loop
    # Concurrency per spec: max_workers=50 for all networking
    return await loop.run_in_executor(None, lambda: upload_file_concurrency(file_bytes, mime_type, display_name, 50))


async def upload_file_with_retry(
    file_bytes: bytes,
    mime_type: str,
    display_name: str = "file",
) -> Optional[str]:
    """Upload a file to Gemini Files API with retry logic across key rotation cycles.

    Retries up to 3 rotation cycles, each trying all healthy keys. Between cycles
    waits with backoff to allow per-minute quotas to recover.
    """
    max_upload_cycles = 3
    for cycle in range(max_upload_cycles):
        result = await upload_file_to_gemini(file_bytes, mime_type, display_name)
        if result:
            return result
        if cycle < max_upload_cycles - 1:
            # Check if we have any keys not on long cooldown; if all are on 60s cooldown, wait a bit
            with _key_cooldown_lock:
                now = _now_ts()
                expiries = [exp for exp in _key_cooldown.values() if exp > now]
                if expiries:
                    wait_for = min(expiries) - now
                    # Wait at most 5s between cycles, or backoff
                    delay = min(max(wait_for + 0.5, compute_backoff_delay(cycle, max_delay=8.0, jitter=True)), 8.0)
                else:
                    delay = compute_backoff_delay(cycle, max_delay=8.0, jitter=True)
            logger.info(f"Upload retry cycle {cycle+1}/{max_upload_cycles} waiting {delay:.1f}s")
            await asyncio.sleep(delay)
    return None


async def delete_gemini_file(file_uri: str) -> bool:
    """Delete a file from the Gemini Files API, trying multiple keys."""
    if not await fetch_api_keys():
        return False

    keys_snapshot = list(api_keys)
    rotator = KeyRotator(_next_index(), keys_snapshot)
    max_attempts = min(len(keys_snapshot), 3)
    for attempt in range(max_attempts):
        key = rotator.get_next_key()
        if key is None:
            key = rotator.get_next_key_allow_cooldown()
        if key is None:
            return False
        client = get_genai_client(key)
        try:
            client.files.delete(name=file_uri)
            return True
        except Exception as exc:
            err_str = str(exc).lower()
            logger.warning(f"Failed to delete Gemini file {file_uri} key ...{key[-4:]} attempt {attempt+1}: {exc}")
            if "401" in err_str or "403" in err_str or "404" in err_str:
                # auth errors may be key-specific, rotate; 404 means file not found anyway success?
                if "404" in err_str:
                    return True
                mark_key_error(key, cooldown_seconds=10.0)
                continue
            if "429" in err_str or "503" in err_str or "500" in err_str:
                mark_key_error(key, cooldown_seconds=10.0)
                continue
            return False
    return False


def normalize_mime_type(mime: str) -> str:
    mime = (mime or "").strip().lower()
    gemini_supported = {
        "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/gif",
        "audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/opus", "audio/flac", "audio/aac", "audio/webm", "audio/m4a",
        "video/mp4", "video/webm", "video/quicktime", "video/x-matroska", "video/x-msvideo", "video/3gpp",
        "application/pdf", "text/plain", "text/html", "text/css", "text/javascript", "text/csv", "text/xml", "application/json", "text/markdown",
    }
    if mime in gemini_supported:
        return mime
    if mime.startswith("text/") or "javascript" in mime or "json" in mime or "xml" in mime:
        return "text/plain"
    return "text/plain"
