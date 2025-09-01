from typing import Optional
import numpy as np

def enhance_with_audioldm(
    audio: np.ndarray,
    sr: int = 32000,
    prompt: str = "studio quality jazz trio, warm, clean mix",
    model_id: str = "cvssp/audioldm2",
    steps: int = 25,
    guidance: float = 2.0,
    device: Optional[str] = None,
):
    try:
        import torch
        from diffusers import AudioLDMPipeline
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pipe = AudioLDMPipeline.from_pretrained(model_id)
        pipe = pipe.to(device)
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                audio=audio,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).audios[0]
        # Ensure output is numpy float32
        return np.asarray(out, dtype=np.float32), sr
    except Exception:
        # Fallback: no change
        return np.asarray(audio, dtype=np.float32), sr
