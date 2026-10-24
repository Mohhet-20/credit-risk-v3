"""Core cross-cutting utilities (config, logging, exceptions, constants).

Importing this package eagerly applies the XGBoost/scikit-learn tags
compatibility patch (see `core.sklearn_compat`) so every downstream module
that touches `core.*` gets the fix for free without remembering to import it
explicitly.
"""

from __future__ import annotations

from core import sklearn_compat as _sklearn_compat  # noqa: F401  (side-effect import)
