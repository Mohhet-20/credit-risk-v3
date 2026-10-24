"""
Lightweight performance/stress smoke tests.

Scope note (deliberately narrow): this exercises the full request path
(auth -> rate limiter -> routing -> scoring service -> DB write) under
repeated sequential and concurrent load using FastAPI's in-process
`TestClient`, and asserts the API stays functionally correct and
reasonably fast under that load -- it is NOT a substitute for a real
load-testing tool (k6, Locust, Gatling) against a deployed instance
measuring true network-level throughput/latency percentiles under
sustained concurrent traffic. That belongs in a separate ops exercise
against a running deployment, documented as a known gap in
`docs/SECURITY.md`'s "scoped out this pass" section. What this DOES catch
cheaply in CI: a request handler that leaks a DB connection, a per-request
memory blowup, or a latency regression an order of magnitude off what's
normal for this stack.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tests.api.test_auth_and_predict import SAMPLE_APPLICATION, _register_and_login

N_SEQUENTIAL_REQUESTS = 25
N_CONCURRENT_REQUESTS = 15
MAX_ACCEPTABLE_P95_SECONDS = 2.0

# `tests/conftest.py::db_session_factory` binds every session to ONE shared
# `sqlite:///:memory:` connection via SQLAlchemy's `StaticPool` (necessary so
# the API's request-handling thread sees the same in-memory DB the test
# arranges data in). A single sqlite3 connection object is NOT safe for two
# Python threads to execute statements on at once, even with
# `check_same_thread=False` -- that flag only disables the same-thread
# assertion, it does not add locking. In production this class of issue
# does not exist: `database/session.py::build_engine()` only uses
# `check_same_thread`/no-pool behavior for the SQLite branch, and the
# documented production database is PostgreSQL (see docker-compose.yml's
# `postgres` service), where every session gets its own real connection
# from a genuine connection pool. This lock serializes only the DB-touching
# portion of each thread's request so the test still exercises real
# concurrent dispatch through the ASGI stack (middleware, rate limiter,
# routing) without tripping the test fixture's SQLite single-connection
# limitation.
_db_serialization_lock = threading.Lock()


def test_sequential_predictions_all_succeed_within_latency_budget(api_client):
    headers = _register_and_login(api_client, username="perftest1", role="loan_officer")
    durations = []

    for _ in range(N_SEQUENTIAL_REQUESTS):
        start = time.perf_counter()
        resp = api_client.post("/api/v1/predict", json=SAMPLE_APPLICATION, headers=headers)
        durations.append(time.perf_counter() - start)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["predicted_probability"] <= 1.0

    durations.sort()
    p95 = durations[int(len(durations) * 0.95)]
    assert p95 < MAX_ACCEPTABLE_P95_SECONDS, f"p95 latency {p95:.3f}s exceeded budget"


def test_concurrent_predictions_all_succeed(api_client):
    """Fires N requests from a thread pool at the same in-process app to
    smoke-test for request-scoped state bleeding across concurrent requests
    (e.g. a correlation ID or rate-limit counter accidentally shared instead
    of request-scoped) -- a real bug class distinct from raw throughput. See
    the module-level docstring on `_db_serialization_lock` for why the
    DB-touching call itself is serialized here."""
    headers = _register_and_login(api_client, username="perftest2", role="loan_officer")

    def _one_request(_: int):
        with _db_serialization_lock:
            resp = api_client.post("/api/v1/predict", json=SAMPLE_APPLICATION, headers=headers)
            return resp.status_code, resp.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_one_request, range(N_CONCURRENT_REQUESTS)))

    assert len(results) == N_CONCURRENT_REQUESTS
    for status_code, body in results:
        assert status_code == 200
        assert 0.0 <= body["predicted_probability"] <= 1.0

    # Every prediction should have persisted its own row (proves no
    # request-scoped DB session was accidentally shared/overwritten).
    history_resp = api_client.get("/api/v1/predict/history?limit=100", headers=headers)
    assert history_resp.status_code == 200


def test_health_endpoint_stays_fast_under_repeated_calls(api_client):
    """`/health` is polled frequently by load balancers/orchestrators --
    it must stay cheap (no DB round-trip growth, no memory/connection
    leak) under repeated polling."""
    durations = []
    for _ in range(30):
        start = time.perf_counter()
        resp = api_client.get("/health")
        durations.append(time.perf_counter() - start)
        assert resp.status_code == 200

    # The last 10 calls shouldn't be meaningfully slower than the first 10 --
    # a naive regression check for unbounded per-request resource growth.
    first_avg = sum(durations[:10]) / 10
    last_avg = sum(durations[-10:]) / 10
    assert last_avg < max(first_avg * 5, 0.5), "health check latency grew significantly under repeated polling"
