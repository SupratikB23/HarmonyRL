"""
End-to-end inference:
- Loads the best available checkpoint (RL-finetuned if present, else supervised).
- Samples a token sequence, renders to MIDI and WAV.
- Optional post-process with Diffusers (AudioLDM2).
"""
import os
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from harmonyrl.midi_utils import vocab_size, tokens_to_midi, synth_audio
from harmonyrl.models.lstm import LSTMModel
from harmonyrl.postprocess_diffusers import enhance_with_audioldm

def generate(
    ckpt_dir: str = "checkpoints",
    out_dir: str = "outputs",
    max_new_tokens: int = 1024,
    temperature: float = 0.9,
    top_p: float = 0.95,
    sr: int = 32000,
    use_diffusers: bool = False,
    diffusers_prompt: str = "studio quality jazz trio, warm, clean mix",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = LSTMModel(vocab_size(), embed_dim=512, hidden=768, layers=3, dropout=0.2).to(device)
    ckpt_rl = os.path.join(ckpt_dir, "lstm_rl.pt")
    ckpt_sup = os.path.join(ckpt_dir, "lstm_supervised.pt")
    ckpt = ckpt_rl if os.path.exists(ckpt_rl) else ckpt_sup
    if not os.path.exists(ckpt):
        raise FileNotFoundError("No checkpoint found. Train first (supervised or RL).")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # Generate tokens → MIDI → audio
    tokens = model.sample(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, device=device)
    pm = tokens_to_midi(tokens)
    midi_path = os.path.join(out_dir, "harmonyrl.mid")
    pm.write(midi_path)

    audio = synth_audio(pm, sr=sr)
    audio = np.asarray(audio, dtype=np.float32)

    # Optional diffusers polish
    if use_diffusers:
        audio, _ = enhance_with_audioldm(audio, sr=sr, prompt=diffusers_prompt)

    wav_path = os.path.join(out_dir, "harmonyrl.wav")
    sf.write(wav_path, audio, sr)
    return wav_path

if __name__ == "__main__":
    path = generate()
    print("Saved:", path)
