from typing import Optional

import numpy as np

from harmonyrl.utils.logging import get_logger

log = get_logger("diffusers")


def enhance_with_audioldm(audio: np.ndarray, sr: int = 32000,
                          prompt: str = "studio quality solo piano, warm, clean mix",
                          model_id: str = "cvssp/audioldm2", steps: int = 50,
                          guidance: float = 3.5, device: Optional[str] = None):
    """Text-conditioned audio pass. AudioLDM2 is text-to-audio: it re-renders at the
    same length rather than conditioning on the input, so treat it as a timbre pass."""
    try:
        import torch
        from diffusers import AudioLDM2Pipeline
    except ImportError:
        log.warning("diffusers/torch unavailable; returning audio unchanged")
        return np.asarray(audio, dtype=np.float32), sr

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipe = AudioLDM2Pipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        out = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance,
                   audio_length_in_s=len(audio) / sr).audios[0]
    except Exception as e:
        log.warning(f"AudioLDM2 pass failed ({e}); returning audio unchanged")
        return np.asarray(audio, dtype=np.float32), sr
    return np.asarray(out, dtype=np.float32), 16000
