from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from harmonyrl.midi_utils import (BAR, is_duration, is_pitch, token_to_duration,
                                  token_to_pitch)

# Unison is neutral, not consonant: rewarding it makes "repeat one note" the optimum.
_CONSONANCE = {0: 0.0, 1: -0.5, 2: -0.2, 3: 1.0, 4: 1.0, 5: 0.8,
               6: -0.5, 7: 1.0, 8: 1.0, 9: 1.0, 10: -0.2, 11: -0.5}
_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_TARGET_DENSITY = (2.0, 24.0)


def _pitches(tokens: Sequence[int]) -> List[int]:
    return [token_to_pitch(t) for t in tokens if is_pitch(t)]


def _durations(tokens: Sequence[int]) -> List[int]:
    return [token_to_duration(t) for t in tokens if is_duration(t)]


def _norm_entropy(values: Sequence) -> float:
    if len(values) < 2:
        return 0.0
    counts = np.array(list(Counter(values).values()), dtype=float)
    if len(counts) < 2:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def reward_harmony(tokens: Sequence[int]) -> float:
    p = _pitches(tokens)
    if len(p) < 2:
        return 0.0
    return float(np.mean([_CONSONANCE[abs(a - b) % 12] for a, b in zip(p, p[1:])]))


def reward_scale(tokens: Sequence[int]) -> float:
    """Best-fitting major scale coverage -- rewards staying in a key."""
    p = _pitches(tokens)
    if len(p) < 4:
        return 0.0
    hist = np.bincount([x % 12 for x in p], minlength=12)
    fits = [hist[[(d + s) % 12 for s in _MAJOR]].sum() for d in range(12)]
    return float(max(fits) / len(p))


def reward_rhythm(tokens: Sequence[int]) -> float:
    """Simple metric ratios between adjacent durations, scaled by duration variety."""
    d = _durations(tokens)
    if len(d) < 2:
        return 0.0
    ok = [1.0 if min(a, b) and max(a, b) % min(a, b) == 0 else 0.0 for a, b in zip(d, d[1:])]
    return float(np.mean(ok)) * _norm_entropy(d)


def reward_diversity(tokens: Sequence[int], n: int = 4) -> float:
    """Distinct n-gram ratio over pitches -- collapses to 0 on looped output."""
    p = _pitches(tokens)
    if len(p) < n + 1:
        return 0.0
    grams = [tuple(p[i:i + n]) for i in range(len(p) - n + 1)]
    return len(set(grams)) / len(grams)


def reward_density(tokens: Sequence[int]) -> float:
    n = len(_pitches(tokens))
    if n == 0:
        return 0.0
    bars = max(1, sum(1 for t in tokens if t == BAR))
    per_bar = n / bars
    lo, hi = _TARGET_DENSITY
    if lo <= per_bar <= hi:
        return 1.0
    return float(np.exp(-abs(per_bar - (lo if per_bar < lo else hi)) / hi))


def reward_parts(tokens: Sequence[int]) -> Dict[str, float]:
    return {
        "harmony": reward_harmony(tokens),
        "scale": reward_scale(tokens),
        "rhythm": reward_rhythm(tokens),
        "diversity": reward_diversity(tokens),
        "density": reward_density(tokens),
    }


def combine_rewards(parts: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(w * parts.get(k, 0.0) for k, w in weights.items()))


def total_reward(tokens: Sequence[int], weights: Dict[str, float]) -> float:
    return combine_rewards(reward_parts(tokens), weights)
