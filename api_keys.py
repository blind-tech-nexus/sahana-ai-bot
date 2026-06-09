import httpx
import logging
from typing import Optional
from config import POOL_API

logger = logging.getLogger("mero.api_keys")

# Global variables to store API keys and their statuses across request cycles
api_keys: list[str] = []
key_status: dict[str, str] = {}  # key -> "unused" | "success" | "failed"


async def fetch_api_keys() -> bool:
    """Fetch API keys from the pool API and update the global tracker."""
    global api_keys, key_status
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(POOL_API)
            if resp.status_code == 200:
                keys = resp.json()
                if isinstance(keys, list) and keys:
                    new_keys = [k for k in keys if k and isinstance(k, str)]
                    api_keys = new_keys

                    # Sync global status dictionary
                    updated_status = {}
                    for k in new_keys:
                        # Retain existing status if key was already present
                        updated_status[k] = key_status.get(k, "unused")
                    key_status = updated_status

                    # If all keys are marked "failed", reset them to "unused" to avoid lockouts
                    if new_keys and all(key_status[k] == "failed" for k in new_keys):
                        for k in key_status:
                            key_status[k] = "unused"

                    logger.info("Fetched %d api keys. Status tracker updated.", len(api_keys))
                    return bool(api_keys)
    except Exception as exc:
        logger.warning("fetch_api_keys failed: %s", exc)
    return bool(api_keys)


def get_keys() -> list[str]:
    """Get the current list of API keys, ensuring they are tracked in key_status."""
    global key_status
    for k in api_keys:
        if k not in key_status:
            key_status[k] = "unused"
    return api_keys


class KeyRotator:
    """Manages sequential key iteration and state tracking for a single request sequence."""

    def __init__(self, preferred_key: Optional[str] = None):
        # Ensure status dictionary has all current keys initialized
        all_keys = list(get_keys())

        # Group keys by their global status
        success_keys = [k for k in all_keys if key_status.get(k) == "success"]
        unused_keys = [k for k in all_keys if key_status.get(k) == "unused"]
        failed_keys = [k for k in all_keys if key_status.get(k) == "failed"]

        # Order of execution: preferred key -> success keys -> unused keys -> failed keys (fallback)
        order = []
        if preferred_key and preferred_key in all_keys:
            order.append(preferred_key)

        for k in success_keys:
            if k not in order:
                order.append(k)
        for k in unused_keys:
            if k not in order:
                order.append(k)
        for k in failed_keys:
            if k not in order:
                order.append(k)

        self._keys_to_try = order
        self._tried_in_request = set()
        self._current_index = 0
        self._failures = []

    @property
    def total_keys(self) -> int:
        return len(self._keys_to_try)

    @property
    def tried_keys(self) -> dict[str, str]:
        # Return a copy of global key_status for debugging/logging
        return dict(key_status)

    @property
    def remaining_keys(self) -> list[str]:
        return [k for k in self._keys_to_try if k not in self._tried_in_request]

    def get_next_key(self) -> Optional[str]:
        """Return the next untried API key for this request cycle. Never repeats keys."""
        while self._current_index < len(self._keys_to_try):
            key = self._keys_to_try[self._current_index]
            self._current_index += 1
            if key not in self._tried_in_request:
                self._tried_in_request.add(key)
                return key
        return None

    def mark_success(self, key: str) -> None:
        """Mark a key as successfully working globally."""
        global key_status
        if key in key_status:
            key_status[key] = "success"
        logger.info("API Key succeeded: %s...%s", key[:6] if len(key) > 6 else "", key[-4:] if len(key) > 4 else "")

    def mark_failed(self, key: str, reason: str = "") -> None:
        """Mark a key as failed globally."""
        global key_status
        if key in key_status:
            key_status[key] = "failed"

        idx = api_keys.index(key) + 1 if key in api_keys else 0
        failure_msg = f"key#{idx}: {reason}" if reason else f"key#{idx}: failed"
        self._failures.append(failure_msg)
        logger.warning(
            "API Key failed: %s...%s (index: %d) Reason: %s",
            key[:6] if len(key) > 6 else "",
            key[-4:] if len(key) > 4 else "",
            idx,
            reason,
        )

    def get_failure_summary(self) -> str:
        """Return a formatted string summarizing the failures of all tried keys."""
        if not self._failures:
            return "No keys tried or available"
        return "; ".join(self._failures)

    def has_keys(self) -> bool:
        """Return True if there is at least one key available to try."""
        return len(self._keys_to_try) > 0
