"""
Unit tests for `core/cache.py`: the in-memory TTL cache (default backend),
the Redis-backed backend (exercised against `fakeredis`, never a live
Redis server -- see `docs/DOCKER_COMPOSE.md` for why this sandbox/CI don't
run one), and the `get_cache()` factory's Redis-vs-in-memory selection
logic including its "fall back on connection failure" behavior.
"""

from __future__ import annotations

import time

import fakeredis
import pytest

import core.cache as cache_mod
from core.cache import InMemoryTTLCache, RedisCache, cached, get_cache, reset_cache_for_testing


@pytest.fixture(autouse=True)
def _reset_cache_singleton():
    """Every test gets a clean `get_cache()` singleton -- without this,
    whichever backend the first test resolves would leak into every test
    that runs after it in the same pytest process."""
    reset_cache_for_testing()
    yield
    reset_cache_for_testing()


# --- InMemoryTTLCache -------------------------------------------------------


def test_in_memory_cache_set_then_get():
    cache = InMemoryTTLCache()
    cache.set("k", {"a": 1}, ttl_seconds=60)
    assert cache.get("k") == {"a": 1}


def test_in_memory_cache_miss_returns_none():
    cache = InMemoryTTLCache()
    assert cache.get("missing") is None


def test_in_memory_cache_expires_after_ttl():
    cache = InMemoryTTLCache()
    cache.set("k", "v", ttl_seconds=0.01)
    time.sleep(0.05)
    assert cache.get("k") is None


def test_in_memory_cache_delete():
    cache = InMemoryTTLCache()
    cache.set("k", "v", ttl_seconds=60)
    cache.delete("k")
    assert cache.get("k") is None


def test_in_memory_cache_delete_prefix():
    cache = InMemoryTTLCache()
    cache.set("model:v1", "a", ttl_seconds=60)
    cache.set("model:v2", "b", ttl_seconds=60)
    cache.set("other:v1", "c", ttl_seconds=60)
    cache.delete_prefix("model:")
    assert cache.get("model:v1") is None
    assert cache.get("model:v2") is None
    assert cache.get("other:v1") == "c"


def test_in_memory_cache_clear():
    cache = InMemoryTTLCache()
    cache.set("a", 1, ttl_seconds=60)
    cache.set("b", 2, ttl_seconds=60)
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_cached_decorator_only_calls_wrapped_function_once():
    calls = []

    @cached(key_fn=lambda x: f"square:{x}", ttl_seconds=60)
    def square(x):
        calls.append(x)
        return x * x

    assert square(4) == 16
    assert square(4) == 16
    assert calls == [4]  # second call was served from cache


# --- RedisCache (against fakeredis, never a live server) -------------------


class _FakeRedisCache(RedisCache):
    """Same class under test, but with `__init__` swapped to use a
    `fakeredis` client instead of opening a real TCP connection -- this
    exercises every line of `RedisCache`'s get/set/delete/delete_prefix/clear
    logic (JSON round-tripping, key namespacing, SCAN-based prefix deletion)
    without needing a Redis server available in this sandbox or in CI."""

    def __init__(self, key_prefix: str = "test:"):
        self._client = fakeredis.FakeStrictRedis(decode_responses=True)
        self._prefix = key_prefix


def test_redis_cache_set_then_get_round_trips_json():
    cache = _FakeRedisCache()
    cache.set("k", {"nested": [1, 2, 3]}, ttl_seconds=60)
    assert cache.get("k") == {"nested": [1, 2, 3]}


def test_redis_cache_miss_returns_none():
    cache = _FakeRedisCache()
    assert cache.get("missing") is None


def test_redis_cache_delete():
    cache = _FakeRedisCache()
    cache.set("k", "v", ttl_seconds=60)
    cache.delete("k")
    assert cache.get("k") is None


def test_redis_cache_delete_prefix_only_removes_matching_keys():
    cache = _FakeRedisCache()
    cache.set("model:v1", "a", ttl_seconds=60)
    cache.set("model:v2", "b", ttl_seconds=60)
    cache.set("other:v1", "c", ttl_seconds=60)
    cache.delete_prefix("model:")
    assert cache.get("model:v1") is None
    assert cache.get("model:v2") is None
    assert cache.get("other:v1") == "c"


def test_redis_cache_keys_are_namespaced_on_the_wire():
    cache = _FakeRedisCache(key_prefix="myapp:")
    cache.set("k", "v", ttl_seconds=60)
    # The raw key on the wire carries the prefix -- this is what makes
    # `delete_prefix` safe to scope to just this app's keys in a Redis
    # instance that might be shared with other consumers.
    assert cache._client.get("myapp:k") is not None
    assert cache._client.get("k") is None


def test_redis_cache_ping():
    cache = _FakeRedisCache()
    assert cache.ping() is True


# --- get_cache() factory selection logic ------------------------------------


def test_get_cache_defaults_to_in_memory_when_redis_url_unset(monkeypatch):
    from configs.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "redis_url", None)
    monkeypatch.setattr(cache_mod, "get_settings", lambda: settings, raising=False)

    cache = get_cache()
    assert isinstance(cache, InMemoryTTLCache)


def test_get_cache_falls_back_to_in_memory_when_redis_unreachable(monkeypatch):
    from configs.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "redis_url", "redis://this-host-does-not-exist.invalid:6379/0")

    class _PatchedSettingsModule:
        @staticmethod
        def get_settings():
            return settings

    monkeypatch.setattr("configs.settings.get_settings", lambda: settings)

    cache = get_cache()
    # A genuinely unreachable Redis URL must not crash `get_cache()` --
    # it degrades to the in-memory backend so the app can still start.
    assert isinstance(cache, InMemoryTTLCache)


def test_get_cache_is_memoized_across_calls(monkeypatch):
    from configs.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "redis_url", None)
    monkeypatch.setattr("configs.settings.get_settings", lambda: settings)

    first = get_cache()
    second = get_cache()
    assert first is second
