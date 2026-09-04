import os
from typing import Optional

import numpy as np
import soundfile as sf

from harmonyrl.midi_utils import EOS, synth_audio, tokens_to_midi
from harmonyrl.utils import device_of, evaluate_tokens, get_logger, load_model


def find_checkpoint(ckpt_dir: str) -> str:
    for name in ("transformer_rl.pt", "lstm_rl.pt",
                 "transformer_supervised.pt", "lstm_supervised.pt"):
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint in {ckpt_dir}; train first.")


def generate(ckpt: Optional[str] = None, ckpt_dir: str = "checkpoints", out_dir: str = "outputs",
             n_samples: int = 1, max_new_tokens: int = 1024, temperature: float = 0.95,
             top_p: float = 0.95, tempo: float = 120.0, sr: int = 32000,
             render_audio: bool = True, use_diffusers: bool = False,
             diffusers_prompt: str = "studio quality solo piano, warm, clean mix"):
    log = get_logger("inference")
    os.makedirs(out_dir, exist_ok=True)
    device = device_of()

    model, meta = load_model(ckpt or find_checkpoint(ckpt_dir), device)
    seq = model.sample(batch_size=n_samples, max_new_tokens=max_new_tokens,
                       temperature=temperature, top_p=top_p, device=device)

    paths = []
    for i, row in enumerate(seq.tolist()):
        tokens = row[: row.index(EOS) + 1] if EOS in row else row
        metrics = evaluate_tokens(tokens)
        log.info(f"sample {i}: " + " ".join(f"{k}={v:.2f}" for k, v in metrics.items()))

        pm = tokens_to_midi(tokens, tempo=tempo)
        midi_path = os.path.join(out_dir, f"harmonyrl_{i}.mid")
        pm.write(midi_path)
        paths.append(midi_path)

        if render_audio and pm.instruments[0].notes:
            audio = np.asarray(synth_audio(pm, sr=sr), dtype=np.float32)
            out_sr = sr  # keep `sr` intact: the enhancer resamples, and the next
            if use_diffusers:  # sample must still be synthesised at the requested rate
                from harmonyrl.postprocess_diffusers import enhance_with_audioldm
                audio, out_sr = enhance_with_audioldm(audio, sr=sr, prompt=diffusers_prompt)
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.9
            sf.write(os.path.join(out_dir, f"harmonyrl_{i}.wav"), audio, out_sr)

    return paths


if __name__ == "__main__":
    print("Saved:", generate())
