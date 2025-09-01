from typing import List
from harmonyrl.midi_utils import token_is_pitch, token_to_pitch

_CONS = {0,3,4,5,7,8,9}
def _consonance(a: int, b: int) -> float:
    iv = abs(a - b) % 12
    return 1.0 if iv in _CONS else -0.5

def simple_harmony_reward(tokens: List[int]) -> float:
    pitches = [token_to_pitch(t) for t in tokens if token_is_pitch(t)]
    if len(pitches) < 2: return 0.0
    score = 0.0
    for a, b in zip(pitches, pitches[1:]): score += _consonance(a, b)
    return score / (len(pitches) - 1)