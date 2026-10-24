"""
Shared `slowapi.Limiter` instance.

Design decision: the limiter used to be constructed inline in `api/app.py`.
Pulling it out here lets router modules (`api/routers/auth.py`,
`api/routers/scoring.py`) apply a *stricter* route-level limit on top of the
global default -- `@limiter.limit(...)` needs the same `Limiter` instance
that's registered on `app.state.limiter`, and routers importing it from
`api/app.py` directly would create a circular import (`api/app.py` is what
imports and mounts the routers in the first place).

Design decision (this pass): the key function is proxy-aware, not raw
`get_remote_address`. This platform's documented deployment topology
(`deployment/kubernetes/07-ingress.yaml`, `docker-compose.yml`'s frontend/
backend network split) always puts a reverse proxy/ingress in front of the
API, so `request.client.host` is the proxy's IP for every request -- every
client would share one global rate-limit bucket instead of being limited
individually, which either starves everyone after one bad actor's traffic
or (if the proxy round-robins across replicas) doesn't limit anyone
meaningfully. Preferring `X-Forwarded-For`'s first hop (the original
client, by convention) fixes that; falling back to `get_remote_address`
keeps local dev and direct-connection deployments working unchanged.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from configs.settings import get_settings

settings = get_settings()


def get_client_ip(request: Request) -> str:
    """Rate-limiting key: the first hop of `X-Forwarded-For` if present
    (the real client, per convention, when a trusted reverse proxy/ingress
    appends its own hop after it), otherwise the raw socket address."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=get_client_ip, default_limits=[f"{settings.rate_limit_per_minute}/minute"])
