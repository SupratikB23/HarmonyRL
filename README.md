# HarmonyRL

Symbolic music generation with an LSTM/Transformer trained on MIDI (MAESTRO/Lakh), plus RL fine-tuning
with a simple music-theory reward. Clean package layout, runnable via `python -m scripts.train_*`.

## Quickstart
```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
# put some .mid/.midi files into data/maestro
python -m scripts.train_supervised
python -m scripts.train_rl