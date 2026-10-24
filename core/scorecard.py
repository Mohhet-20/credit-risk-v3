"""
PD -> scorecard points (the "credit score" a loan officer actually reads).

Standard points-to-double-odds (PDO) scaling used by bureau and in-house
scorecards: a base score is pinned to a base good:bad odds, and every PDO
points the odds double. With the defaults below, 600 points means 50:1
good:bad odds (PD ~1.96%), 620 means 100:1, 580 means 25:1.

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln((1 - PD) / PD)

Scores are clipped to the familiar 300-850 range. Because the mapping is
strictly monotone in PD, it changes nothing about ranking or decisions --
it is a presentation layer that makes PD legible to non-quants.
"""

from __future__ import annotations

import math

BASE_SCORE = 600
BASE_ODDS = 50.0
PDO = 20
MIN_SCORE, MAX_SCORE = 300, 850

_FACTOR = PDO / math.log(2)
_OFFSET = BASE_SCORE - _FACTOR * math.log(BASE_ODDS)


def pd_to_score(pd_value: float) -> int:
    p = min(max(pd_value, 1e-6), 1 - 1e-6)
    score = _OFFSET + _FACTOR * math.log((1 - p) / p)
    return int(round(min(max(score, MIN_SCORE), MAX_SCORE)))


def score_to_pd(score: float) -> float:
    odds = math.exp((score - _OFFSET) / _FACTOR)
    return 1.0 / (1.0 + odds)
