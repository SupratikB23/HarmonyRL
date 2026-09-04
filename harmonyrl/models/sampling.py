import torch

from harmonyrl.midi_utils import BOS, PAD

# Structural tokens the model must never emit mid-stream: PAD only ever appears as
# right-padding in training targets (it is ignored by the loss, so its logit is
# unconstrained) and BOS only ever appears at position 0.
FORBIDDEN = (PAD, BOS)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Drop the tail of the distribution, always keeping the top token."""
    if not 0.0 < top_p < 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(-1)
    drop_sorted = (probs.cumsum(-1) - probs) > top_p
    drop_sorted[..., 0] = False
    drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
    return logits.masked_fill(drop, float("-inf"))


def sample_step(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    logits = logits / max(1e-6, temperature)  # new tensor; safe to mask in place
    for tok in FORBIDDEN:
        logits[..., tok] = float("-inf")
    return torch.multinomial(top_p_filter(logits, top_p).softmax(-1), 1)
