from collections import Counter
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from harmonyrl.midi_utils import PAD, is_pitch, token_to_pitch
from harmonyrl.rewards import reward_parts


def _pitches(tokens: Sequence[int]):
    return [token_to_pitch(t) for t in tokens if is_pitch(t)]


def distinct_n(tokens: Sequence[int], n: int = 4) -> float:
    p = _pitches(tokens)
    if len(p) < n + 1:
        return 0.0
    grams = [tuple(p[i:i + n]) for i in range(len(p) - n + 1)]
    return len(set(grams)) / len(grams)


def max_repeat_run(tokens: Sequence[int]) -> int:
    """Longest run of one repeated pitch -- the classic degenerate-policy signature."""
    p = _pitches(tokens)
    best = run = 1 if p else 0
    for a, b in zip(p, p[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def pitch_class_entropy(tokens: Sequence[int]) -> float:
    p = _pitches(tokens)
    if not p:
        return 0.0
    counts = np.array(list(Counter(x % 12 for x in p).values()), dtype=float)
    q = counts / counts.sum()
    return float(-(q * np.log2(q)).sum())


@torch.no_grad()
def perplexity(model, loader, device: str = "cpu") -> float:
    model.eval()
    total, count = 0.0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)[0]
        n = int((Y != PAD).sum())
        if n == 0:
            continue
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), Y.reshape(-1),
                                 ignore_index=PAD).item() * n
        count += n
    return float(np.exp(total / max(1, count)))


def evaluate_tokens(tokens: Sequence[int]) -> Dict[str, float]:
    out = reward_parts(tokens)
    out.update(n_notes=float(len(_pitches(tokens))),
               distinct_4=distinct_n(tokens),
               max_repeat_run=float(max_repeat_run(tokens)),
               pitch_class_entropy=pitch_class_entropy(tokens))
    return out
