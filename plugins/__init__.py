"""
Modular plugin architecture: a lightweight namespace for self-contained,
independently maintainable extension modules that plug into the core
platform's workflow (`workflows/engine.py`) without the core platform
importing anything plugin-SPECIFIC -- only a small, stable interface each
plugin implements.

Design decision (important, and deliberately conservative): this is
ADDITIVE. No existing router, service, or repository was moved into
`plugins/` or refactored to route through a plugin registry -- the
instruction for this pass was explicit ("DO NOT rewrite existing
modules"), and a mature, working, fully-tested codebase is exactly the
wrong place to introduce a speculative "everything is a plugin"
re-architecture for its own sake. What `plugins/` provides instead is the
PATTERN for genuinely new, optional capability going forward:
`plugins/fraud_detection/` is the first real example (a documented
placeholder today -- see its module docstring -- with the exact interface
a real fraud-detection integration would implement tomorrow). Future
candidates explicitly named in the platform's own roadmap (a real
LLM-backed Copilot mode, additional model families, alternative
notification providers) are natural fits for this same pattern without
requiring `api/routers/`, `services/`, or `workflows/engine.py` to change
shape again each time one is added.

Each plugin is a plain Python package under `plugins/<name>/` exposing
whatever functions the core platform's extension points call (e.g.
`workflows/engine.py`'s fraud-check step calls
`plugins.fraud_detection.run_fraud_check()`). There is intentionally no
dynamic plugin-discovery/loading machinery (no entry-points scanning, no
plugin manifest format) -- for a platform with a handful of first-party
extension points, explicit imports at the one or two call sites that need
them are simpler to read, debug, and type-check than a generic loader
would be, and adding real dynamic discovery is a reasonable future step
if/when third-party (not just first-party) plugins become a real need.
"""

from __future__ import annotations
