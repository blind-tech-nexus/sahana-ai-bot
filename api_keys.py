import httpx
import logging
from typing import Optional
from config import POOL_API

logger = logging.getLogger("mero.api_keys")

api_keys: list[str] = []


async def fetch_api_keys() -> bool:
    global api_keys
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(POOL_API)
            if resp.status_code == 200:
                keys = resp.json()
                if isinstance(keys, list) and keys:
                    api_keys = [k for k in keys if k and isinstance(k, str)]
                    logger.info("fetched %d api keys", len(api_keys))
                    return bool(api_keys)
    except Exception as exc:
        logger.warning("fetch_api_keys failed: %s", exc)
    return bool(api_keys)


def get_keys() -> list[str]:
    return api_keys


class KeyRotator:
    def __init__(self, preferred_key: Optional[str] = None):
        all_keys = [k for k in get_keys() if k]
        if preferred_key and preferred_key in all_keys:
            self._keys = [preferred_key] + [k for k in all_keys if k != preferred_key]
        else:
            self._keys = all_keys
        self._status: dict[str, str] = {k: "unused" for k in self._keys}
        self._failures: list[str] = []
        self._current_index = 0

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    @property
    def tried_keys(self) -> dict[str, str]:
        return dict(self._status)

    @property
    def remaining_keys(self) -> list[str]:
        return [k for k, s in self._status.items() if s == "unused"]

    @property
    def failed_keys(self) -> list[str]:
        return [k for k, s in self._status.items() if s == "failed"]

    @property
    def successful_key(self) -> Optional[str]:
        for k, s in self._status.items():
            if s == "success":
                return k
        return None

    def get_next_key(self) -> Optional[str]:
        while self._current_index < len(self._keys):
            key = self._keys[self._current_index]
            if self._status[key] == "unused":
                return key
            self._current_index += 1
        return None

    def mark_success(self, key: str) -> None:
        if key in self._status:
            self._status[key] = "success"
            logger.debug("key #%d succeeded", self._keys.index(key) + 1)

    def mark_failed(self, key: str, reason: str = "") -> None:
        if key in self._status:
            self._status[key] = "failed"
            idx = self._keys.index(key) + 1
            failure_msg = f"key#{idx}: {reason}" if reason else f"key#{idx}: failed"
            self._failures.append(failure_msg)
            logger.debug("key #%d failed: %s", idx, reason)
            self._current_index += 1

    def get_failure_summary(self) -> str:
        if not self._failures:
            return "No keys available"
        return "; ".join(self._failures)

    def has_keys(self) -> bool:
        return len(self._keys) > 0
