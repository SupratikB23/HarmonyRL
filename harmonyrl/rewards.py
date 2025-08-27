"""
Rewards that can be used during RL fine-tuning.

Includes:
- Symbolic harmony/rhythm rewards (fast, no GPU needed)
- Optional HF audio-based rewards (genre/style and CLAP similarity)

These are written to be robust: if HF pipelines/models aren't available,
they fail closed and return 0.0 (so training can still proceed).
"""
from typing import Dict, List, Optional
import numpy as np

from harmonyrl.midi_utils import (
    token_is_pitch, token_to_pitch, token_to_dur
)

# ---------------------------
# Symbolic (token-based) rewards
# ---------------------------

# Unison/3rds/4th/5th/6ths considered consonant
_CONS = {0, 3, 4, 5, 7, 8, 9}

def _consonance(a: int, b: int) -> float:
    iv = abs(a - b) % 12
    return 1.0 if iv in _CONS else -0.5

def reward_harmony(tokens: List[int]) -> float:
    """Consecutive-note consonance."""
    pitches = [token_to_pitch(t) for t in tokens if token_is_pitch(t)]
    if len(pitches) < 2:
        return 0.0
    score = 0.0
    for a, b in zip(pitches, pitches[1:]):
        score += _consonance(a, b)
    return score / (len(pitches) - 1)

def reward_rhythm(tokens: List[int]) -> float:
    """Prefer simple rhythmic ratios (durations close to integers when adjacent)."""
    durs = [token_to_dur(t) for t in tokens if t >= 256]
    if len(durs) < 2:
        return 0.0
    ratios = [durs[i+1] / durs[i] if durs[i] != 0 else 1.0 for i in range(len(durs)-1)]
    penalties = [abs(round(r) - r) for r in ratios]
    return 1.0 - float(np.mean(np.clip(penalties, 0.0, 1.0)))

# ---------------------------
# Audio-based rewards (HuggingFace)
# ---------------------------

_HF_AUDIO_PIPE = None
_HF_CLAP_PIPE = None

def _get_audio_classifier(model_id: str):
    global _HF_AUDIO_PIPE
    if _HF_AUDIO_PIPE is None:
        try:
            from transformers import pipeline
            _HF_AUDIO_PIPE = pipeline("audio-classification", model=model_id, top_k=5)
        except Exception:
            _HF_AUDIO_PIPE = False
    return _HF_AUDIO_PIPE

def _get_clap():
    global _HF_CLAP_PIPE
    if _HF_CLAP_PIPE is None:
        try:
            from transformers import pipeline
            _HF_CLAP_PIPE = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused")
        except Exception:
            _HF_CLAP_PIPE = False
    return _HF_CLAP_PIPE

def reward_style(audio_np, sr: int, style_prompt: str, hf_model_id: str) -> float:
    """
    Uses an audio classifier to check whether the generated audio matches the style prompt.
    Heuristic: checks token overlap between classifier labels and prompt words.
    """
    pipe = _get_audio_classifier(hf_model_id)
    if not pipe:
        return 0.0
    try:
        res = pipe({"array": audio_np, "sampling_rate": sr})
    except Exception:
        return 0.0
    prompt_tokens = set(style_prompt.lower().split())
    hit = 0.0
    for r in res:
        if any(w in r["label"].lower() for w in prompt_tokens):
            hit = max(hit, float(r["score"]))
    return float(hit)

def reward_clap(audio_np, sr: int, text_prompt: str) -> float:
    """CLAP text–audio similarity (zero-shot)."""
    pipe = _get_clap()
    if not pipe:
        return 0.0
    try:
        res = pipe({"array": audio_np, "sampling_rate": sr}, candidate_labels=[text_prompt])
        return float(res[0]["score"]) if res else 0.0
    except Exception:
        return 0.0

def combine_rewards(parts: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(weights.get(k, 0.0) * parts.get(k, 0.0) for k in set(parts) | set(weights)))
