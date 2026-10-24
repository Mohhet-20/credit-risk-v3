"""
Cache backend: in-process TTL cache (default) with a Redis-backed swap-in
for multi-replica deployments.

Design decision: a small dict-based cache with an explicit `delete`/
`delete_prefix`/`clear` API, not `functools.lru_cache`, because `lru_cache`
has no TTL and no way to invalidate a single key from outside the decorated
function -- exactly what "bust the model-comparison cache after a
promotion" requires. `CacheBackend` is intentionally the same shape a
Redis-backed implementation exposes (get/set/delete/clear), so swapping to
Redis for a multi-process/multi-host deployment is a one-file, one-env-var
change -- no call site elsewhere in the codebase needs to change.

Design decision: `get_cache()` picks the backend based on
`settings.redis_url` at first call (memoized in a module-level singleton,
not re-checked every call). If `REDIS_URL` is set but the server is
unreachable at process start, this falls back to the in-memory backend with
a logged warning rather than crashing the whole app on startup -- a cache is
a performance optimization, not a hard dependency, and a redis blip
shouldn't take the API down. `RedisCache` JSON-serializes values (never
pickles) so cached data is inspectable with `redis-cli GET` and can never
execute arbitrary code on read -- every current caller (model-comparison
rows, monitoring dashboard stats) is already JSON-serializable.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, TypeVar

from core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl_seconds: float) -> None: ...
    def delete(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> None: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...


@dataclass
class _Entry:
    value: Any
    expires_at: float


class InMemoryTTLCache:
    """Thread-safe in-process cache -- correct for a single API process
    (local dev, tests, a single-replica deployment); see the module
    docstring for the Redis swap path once the API scales to multiple
    replicas that need a *shared* cache."""

    def __init__(self) -> None:
        self._store: dict[str, _Entry] = {}
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                del self._store[key]
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        with self._lock:
            self._store[key] = _Entry(value=value, expires_at=time.monotonic() + ttl_seconds)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in [k for k in self._store if k.startswith(prefix)]:
                del self._store[key]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def close(self) -> None:
        """No-op: there is no external connection to release for an
        in-process dict cache. Exists so `api/app.py`'s shutdown handler can
        call `get_cache().close()` unconditionally without branching on
        which backend is active."""


class RedisCache:
    """Redis-backed `CacheBackend` -- the swap-in for multi-replica
    deployments described in the module docstring. Requires the `redis`
    package (declared in `requirements.txt`; only imported here, so a
    deployment that never sets `REDIS_URL` never needs the dependency
    importable at all beyond having it installed)."""

    def __init__(self, redis_url: str, key_prefix: str = "credit-risk:", socket_timeout: float = 2.0):
        import redis as redis_lib  # local import: keeps `get_cache()` working even if `redis` were ever made optional

        self._client = redis_lib.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=socket_timeout,
            socket_timeout=socket_timeout,
        )
        self._prefix = key_prefix

    def ping(self) -> bool:
        return bool(self._client.ping())

    def _namespaced(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        raw = self._client.get(self._namespaced(key))
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        self._client.set(self._namespaced(key), json.dumps(value), ex=max(1, int(ttl_seconds)))

    def delete(self, key: str) -> None:
        self._client.delete(self._namespaced(key))

    def delete_prefix(self, prefix: str) -> None:
        pattern = f"{self._namespaced(prefix)}*"
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break

    def clear(self) -> None:
        self.delete_prefix("")

    def close(self) -> None:
        """Releases the underlying Redis connection pool. Called from
        `api/app.py`'s shutdown lifespan handler as part of graceful
        shutdown -- without it, in-flight connections are only reclaimed
        when the process exits and the OS closes the sockets, which is fine
        for a CLI script but sloppy for a long-lived server process that may
        be restarted many times (e.g. a Kubernetes rolling deploy)."""
        self._client.close()


_cache: CacheBackend | None = None
_cache_init_lock = RLock()


def get_cache() -> CacheBackend:
    """Returns the process-wide cache singleton, choosing Redis vs.
    in-memory on first call based on `settings.redis_url` (see module
    docstring). Safe to call before the DB/app is otherwise initialized."""
    global _cache
    if _cache is not None:
        return _cache

    with _cache_init_lock:
        if _cache is not None:
            return _cache

        from configs.settings import get_settings

        settings = get_settings()
        if settings.redis_url:
            try:
                candidate = RedisCache(settings.redis_url, key_prefix=settings.cache_key_prefix)
                candidate.ping()
                _cache = candidate
                logger.info("Cache backend: Redis", extra={"context": {"redis_url_set": True}})
            except Exception as exc:  # noqa: BLE001 -- a cache is best-effort, never fatal at startup
                logger.warning(
                    "Redis cache unreachable, falling back to in-memory cache",
                    extra={"context": {"error": str(exc)}},
                )
                _cache = InMemoryTTLCache()
        else:
            _cache = InMemoryTTLCache()
    return _cache


def reset_cache_for_testing() -> None:
    """Clears the memoized backend choice so tests can force a fresh
    `get_cache()` resolution (e.g. after monkeypatching `settings.redis_url`).
    Not called anywhere in production code paths."""
    global _cache
    with _cache_init_lock:
        _cache = None


def cached(key_fn: Callable[..., str], ttl_seconds: float):
    """Decorator caching a function's return value, keyed by `key_fn(*args, **kwargs)`."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = get_cache()
            key = key_fn(*args, **kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            cache.set(key, result, ttl_seconds)
            return result

        return wrapper

    return decorator
