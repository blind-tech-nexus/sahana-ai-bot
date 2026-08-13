import os
import random
import httpx
import logging
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from config import POOL_API

logger = logging.getLogger("mero.api_keys")

api_keys: list[str] = []
LAST_FETCH_TIME: float = 0
CACHE_TTL = 300

_KEY_ERROR_COOLDOWN = 30.0
_key_error_times: dict[str, float] = {}
_key_error_lock = threading.Lock()

# Async lock for fetching/rotating keys
_fetch_lock = asyncio.Lock()


async def fetch_api_keys() -> bool:
    """Fetch API keys from the pool endpoint or environment, caching with TTL.

    Thread-safe: only one fetch at a time per event loop (async lock).
    """
    global api_keys, LAST_FETCH_TIME
    async with _fetch_lock:
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
            logger.warning(f"fetch_api_keys from {POOL_API} failed: {exc}")

        env_keys_str = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEYS") or os.environ.get("GEMINI_API_KEYS")
        if env_keys_str:
            api_keys = [k.strip() for k in env_keys_str.split(",") if k.strip()]
            LAST_FETCH_TIME = current_time
            logger.info(f"Using {len(api_keys)} api keys from environment variables.")
            return bool(api_keys)

        if not api_keys:
            logger.error("No API keys available.")
        return bool(api_keys)


def is_key_on_cooldown(key: str) -> bool:
    """Check if a key is on cooldown due to a recent error."""
    err_time = _key_error_times.get(key)
    if err_time is None:
        return False
    return (time.time() - err_time) < _KEY_ERROR_COOLDOWN


def mark_key_error(key: str) -> None:
    """Mark a key as errored and put it on cooldown."""
    with _key_error_lock:
        _key_error_times[key] = time.time()


def clear_key_error(key: str) -> None:
    """Clear the error cooldown state for a key."""
    with _key_error_lock:
        _key_error_times.pop(key, None)


def get_available_keys() -> list[str]:
    """Return list of keys not currently on cooldown (thread-safe snapshot)."""
    with _key_error_lock:
        error_times = dict(_key_error_times)
    now = time.time()
    return [k for k in api_keys if not ((now - error_times.get(k, 0)) < _KEY_ERROR_COOLDOWN)]


class KeyRotator:
    """Synchronous, thread-safe round-robin key rotator.

    Yields keys skipping any that are on cooldown. Designed for use within
    synchronous/blocking contexts (e.g., ThreadPoolExecutor workers) where
    ``asyncio`` event-loop primitives are not available.

    The rotator starts from a given index in the full key list and advances
    deterministically (modulo key count), skipping keys on cooldown. The
    starting index should be obtained from the async iterator to maintain
    correct rotation state.
    """

    def __init__(self, start_idx: int = 0):
        self._start_idx = start_idx
        self._pos = start_idx
        self._tried = 0

    def _snapshot(self) -> list[str]:
        return get_available_keys()

    def get_next_key(self) -> Optional[str]:
        """Get the next available key, skipping keys on cooldown."""
        all_keys = self._snapshot()
        if not all_keys:
            return None

        n = len(all_keys)
        for _ in range(n):
            key = all_keys[self._pos % n]
            self._pos = (self._pos + 1) % n
            self._tried += 1
            if not is_key_on_cooldown(key):
                return key
        # All keys exhausted/tried
        return None

    @property
    def tried_count(self) -> int:
        return self._tried

    @property
    def available_count(self) -> int:
        return len(self._snapshot())


# Async iterator index state (single event loop). Protected by async lock.
_key_index = 0
_key_lock = asyncio.Lock()


async def get_next_key_index() -> int:
    """Async round-robin: returns index of next key, advancing the shared counter."""
    global _key_index
    async with _key_lock:
        if not api_keys:
            return 0
        idx = _key_index
        _key_index = (_key_index + 1) % len(api_keys)
        return idx


def is_retriable_error(e: Exception) -> bool:
    err_str = str(e).lower()
    non_retriable = {"400", "401", "403", "404", "invalid", "permission", "denied", "malformed", "bad request", "safety"}
    if any(c in err_str for c in non_retriable):
        return False
    retriable = {"429", "500", "502", "503", "504", "resource_exhausted", "unavailable", "connection", "timeout"}
    if any(c in err_str for c in retriable):
        return True
    return True


class HTTPException(Exception):
    """Wraps HTTP errors with status code and error payload for inspection."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


async def _gemini_request(
    url: str,
    payload: dict,
    start_idx: int,
) -> tuple[Optional[dict], Optional[HTTPException]]:
    """Make a single Gemini REST API request, rotating keys on retriable errors.

    Returns (parsed_json, error). Parsed JSON is the raw API response dict.
    Handles key rotation, 429 backoff, and retries internally.
    """
    if not api_keys:
        return None, HTTPException(0, "No API keys available")

    rotator = KeyRotator(start_idx)
    limiter = get_rate_limiter()
    max_retries = min(6, max(3, rotator.available_count))
    attempt = 0
    last_error: Optional[HTTPException] = None

    async with httpx.AsyncClient(
        timeout=120.0,
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200),
    ) as client:
        while attempt < max_retries:
            if not await limiter.acquire("gemini_api", timeout=30.0):
                return None, HTTPException(0, "Rate limiter timeout waiting to acquire token")

            key = rotator.get_next_key()
            if key is None:
                if last_error is None:
                    last_error = HTTPException(0, "All API keys exhausted")
                break

            full_url = f"{url}?key={key}"
            try:
                resp = await client.post(full_url, json=payload, headers={"Content-Type": "application/json"})
                if resp.status_code == 200:
                    clear_key_error(key)
                    try:
                        data = resp.json()
                        return data, None
                    except Exception:
                        return None, HTTPException(resp.status_code, resp.text[:500])

                error_text = resp.text
                logger.warning(
                    f"Gemini API failed (key idx offset {rotator.tried_count - 1}) status {resp.status_code}: {error_text[:500]}"
                )
                last_error = HTTPException(resp.status_code, error_text)

                if resp.status_code == 429:
                    retry_after = get_retry_after(resp)
                    mark_key_error(key)
                    if attempt < max_retries - 1:
                        delay = retry_after if retry_after > 0 else compute_backoff_delay(attempt, max_delay=30.0, jitter=True)
                        logger.info(f"429 on key, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    break

                if resp.status_code >= 500:
                    mark_key_error(key)
                    if attempt < max_retries - 1:
                        delay = compute_backoff_delay(attempt, max_delay=15.0, jitter=True)
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    break

                # 400, 401, 403 - non-retriable
                if resp.status_code in (401, 403):
                    mark_key_error(key)
                break

            except httpx.TimeoutException:
                last_error = HTTPException(0, "Request timed out")
                logger.warning(f"Timeout on API call (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    delay = compute_backoff_delay(attempt, max_delay=10.0, jitter=True)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                break
            except Exception as exc:
                last_error = HTTPException(0, str(exc))
                logger.warning(f"API call exception: {exc}")
                if attempt < max_retries - 1:
                    delay = compute_backoff_delay(attempt, max_delay=10.0, jitter=True)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                break

    return None, last_error or HTTPException(0, "All keys exhausted")


def _sync_gemini_request(
    url: str,
    payload: dict,
    start_idx: int,
) -> tuple[Optional[dict], Optional[HTTPException]]:
    """Synchronous version of _gemini_request for use in ThreadPoolExecutor workers.

    Uses httpx synchronously. Rotates keys on retriable errors.
    """
    if not api_keys:
        return None, HTTPException(0, "No API keys available")

    rotator = KeyRotator(start_idx)
    import time as _time

    max_retries = min(6, max(3, rotator.available_count))
    attempt = 0
    last_error: Optional[HTTPException] = None

    with httpx.Client(timeout=120.0) as client:
        while attempt < max_retries:
            key = rotator.get_next_key()
            if key is None:
                if last_error is None:
                    last_error = HTTPException(0, "All API keys exhausted")
                break

            full_url = f"{url}?key={key}"
            try:
                resp = client.post(full_url, json=payload, headers={"Content-Type": "application/json"})
                if resp.status_code == 200:
                    clear_key_error(key)
                    try:
                        data = resp.json()
                        return data, None
                    except Exception:
                        return None, HTTPException(resp.status_code, resp.text[:500])

                error_text = resp.text
                logger.warning(
                    f"Gemini API sync failed status {resp.status_code}: {error_text[:500]}"
                )
                last_error = HTTPException(resp.status_code, error_text)

                if resp.status_code == 429:
                    retry_after = get_retry_after(resp)
                    mark_key_error(key)
                    if attempt < max_retries - 1:
                        delay = retry_after if retry_after > 0 else compute_backoff_delay(attempt, max_delay=30.0, jitter=True)
                        _time.sleep(delay)
                        attempt += 1
                        continue
                    break

                if resp.status_code >= 500:
                    mark_key_error(key)
                    if attempt < max_retries - 1:
                        delay = compute_backoff_delay(attempt, max_delay=15.0, jitter=True)
                        _time.sleep(delay)
                        attempt += 1
                        continue
                    break

                if resp.status_code in (401, 403):
                    mark_key_error(key)
                break

            except httpx.TimeoutException:
                last_error = HTTPException(0, "Request timed out")
                logger.warning(f"Timeout on sync API call (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    delay = compute_backoff_delay(attempt, max_delay=10.0, jitter=True)
                    _time.sleep(delay)
                    attempt += 1
                    continue
                break
            except Exception as exc:
                last_error = HTTPException(0, str(exc))
                logger.warning(f"Sync API call exception: {exc}")
                if attempt < max_retries - 1:
                    delay = compute_backoff_delay(attempt, max_delay=10.0, jitter=True)
                    _time.sleep(delay)
                    attempt += 1
                    continue
                break

    return None, last_error or HTTPException(0, "All keys exhausted")


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
        # Re-check in case another thread created one
        existing = _genai_clients.get(api_key)
        if existing is not None:
            return existing
        _genai_clients[api_key] = client
    return client


def get_next_genai_client() -> Optional["object"]:
    """Get the next available genai.Client using key rotation (thread-safe)."""
    if not api_keys:
        return None
    rotator = KeyRotator(_get_next_sync_key_index())
    key = rotator.get_next_key()
    if key is None:
        return None
    return get_genai_client(key)


_key_index_sync = 0
_key_index_sync_lock = threading.Lock()


def _get_next_sync_key_index() -> int:
    """Sync round-robin key index for use in synchronous worker threads."""
    global _key_index_sync
    with _key_index_sync_lock:
        if not api_keys:
            return 0
        idx = _key_index_sync
        _key_index_sync = (_key_index_sync + 1) % len(api_keys)
        return idx


def upload_file_concurrency(
    file_contents: bytes,
    mime_type: str,
    display_name: str = "file",
    max_workers: int = 50,
) -> Optional[str]:
    """Upload a single file to Gemini Files API.

    Tries keys in round-robin, with up to ``max_workers`` concurrent upload
    attempts. Returns the file URI (e.g. ``files/abc-123``) on success, or None.
    """
    if not api_keys:
        return None

    mime_type = normalize_mime_type(mime_type)

    def _upload_attempt(key: str) -> Optional[str]:
        client = get_genai_client(key)
        try:
            file_obj = client.files.upload(
                file={"file": ("upload.bin", file_contents), "mime_type": mime_type},
                config={"display_name": display_name[:100]},
            )
            name = getattr(file_obj, "name", None)
            if name and name.startswith("files/"):
                return name
            # Some SDK versions return just the name without prefix
            if name:
                return f"files/{name}" if not name.startswith("files/") else name
            return None
        except Exception as exc:
            err_str = str(exc).lower()
            if "401" in err_str or "403" in err_str or "invalid" in err_str or "permission" in err_str:
                mark_key_error(key)
            logger.warning(f"File upload failed for key (attempt): {exc}")
            return None

    if not api_keys:
        return None

    # Prepare candidate keys respecting cooldown
    available = []
    for k in api_keys:
        if not is_key_on_cooldown(k):
            available.append(k)
        if len(available) >= max_workers:
            break

    if not available:
        # Fall back to all keys (ignore cooldown)
        available = list(api_keys)[:max_workers]

    if not available:
        return None

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_upload_attempt, k): k for k in available}
            for future in futures:
                result = future.result()
                if result:
                    clear_key_error(futures[future])
                    return result
                else:
                    mark_key_error(futures[future])
    except Exception as exc:
        logger.warning(f"Concurrency upload helper error: {exc}")

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
        # For smaller files, inline is handled by caller; return None to signal no upload needed.
        return None

    # For larger files, perform upload via concurrency helper in thread pool.
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, upload_file_concurrency, file_bytes, mime_type, display_name, 50)


async def upload_file_with_retry(
    file_bytes: bytes,
    mime_type: str,
    display_name: str = "file",
) -> Optional[str]:
    """Upload a file to Gemini Files API with retry logic (concurrency-supported)."""
    max_upload_retries = 3
    for attempt in range(max_upload_retries):
        result = await upload_file_to_gemini(file_bytes, mime_type, display_name)
        if result:
            return result
        if attempt < max_upload_retries - 1:
            delay = compute_backoff_delay(attempt, max_delay=10.0, jitter=True)
            await asyncio.sleep(delay)
    return None


async def delete_gemini_file(file_uri: str) -> bool:
    """Delete a file from the Gemini Files API.

    file_uri should be like 'files/abc-123'.
    """
    if not await fetch_api_keys():
        return False

    rotator = KeyRotator(_get_next_sync_key_index())
    key = rotator.get_next_key()
    if key is None:
        return False

    client = get_genai_client(key)
    try:
        client.files.delete(name=file_uri)
        return True
    except Exception as exc:
        logger.warning(f"Failed to delete Gemini file {file_uri}: {exc}")
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
