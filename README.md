# 🎶 HarmonyRL

**Symbolic music generation with Supervised Pretraining + Reinforcement Learning fine-tuning + Diffusion postprocessing.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%20%7C%20Diffusers-yellow?logo=huggingface)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

> **Status:** Active experiment — architectures, reward functions, and training loops are all being iterated on.

---

## What Is This?

HarmonyRL is an experimental framework that treats **music generation as a sequential decision-making problem**. The core idea: first teach a model *what music looks like* via supervised learning on real piano performances, then use reinforcement learning to nudge it toward outputs that sound *musically coherent* — not just statistically likely.

The pipeline has three stages:

```
MAESTRO MIDI corpus
        │
        ▼
┌──────────────────┐
│  Supervised      │  Cross-entropy pretraining on tokenized MIDI sequences
│  Pretraining     │  (LSTM or Transformer backbone)
└────────┬─────────┘
         │  checkpoint
         ▼
┌──────────────────┐
│  RL Fine-tuning  │  REINFORCE with EMA baseline + entropy bonus
│  (Policy Grad.)  │  Reward: harmony consonance + rhythmic regularity
└────────┬─────────┘
         │  MIDI tokens
         ▼
┌──────────────────┐
│  Diffusion       │  AudioLDM2 postprocessing on synthesized audio
│  Postprocessing  │  Softens dissonance, improves timbral quality
└──────────────────┘
```

---

## Motivation

Standard language-model-style training on MIDI learns token distributions but has no direct incentive to produce *harmonically pleasant* music. A model can achieve low cross-entropy while generating atonal noise. RL bridges that gap: define what "good music" means as a reward signal and optimize for it directly.

The challenge is that musical quality is hard to quantify. This project experiments with:
- **Symbolic rewards** computed directly on token sequences (fast, differentiable proxies)
- **Perceptual rewards** via CLAP (Contrastive Language-Audio Pretraining) text–audio similarity

---

## Tokenization Scheme

MIDI is converted into a compact integer vocabulary before training:

| Token range | Meaning |
|---|---|
| `0` | `PAD` |
| `1` | `BOS` (begin of sequence) |
| `2` | `EOS` (end of sequence) |
| `3` | `BAR` (measure boundary) |
| `4` | `REST` |
| `16 – 104` | Pitch tokens (MIDI notes 21–108, piano range) |
| `256 – 260` | Duration tokens (60, 120, 240, 480, 960 ticks @ 480 PPQ) |

Total vocabulary size: **261 tokens** (`VOCAB_SIZE = DUR_BASE + len(DURS)` = 256 + 5). Token IDs 5–15 and 105–255 are intentionally left unused as reserved space — the scheme is kept small to keep models lightweight and training fast during experimentation.

---

## Models

Two backbone architectures are available and interchangeable:

### LSTM (`harmonyrl/models/lstm.py`)

A stacked LSTM with tied input/output embeddings and LayerNorm on the hidden state.

```
Embedding(vocab, d) → LSTM(d→h, L layers) → LayerNorm → Dropout → Linear(h→vocab)
```

Weight tying between the embedding and output projection reduces parameters and often improves generation quality.

### Transformer (`harmonyrl/models/transformer.py`)

A Pre-Norm decoder-only Transformer with **Rotary Positional Encoding (RoPE)** instead of learned or sinusoidal embeddings. RoPE encodes relative position information directly into attention scores, which helps the model generalize to sequence lengths not seen during training.

```
Embedding(vocab, d) → RoPE → N × [PreNorm → MHA → PreNorm → FFN] → LayerNorm → Linear(h→vocab)
```

Both models use **nucleus (top-p) sampling** at inference time.

---

## Reward Design

The RL reward signal is a combination of symbolic and (optionally) perceptual components:

### Harmony Reward

Consecutive note pairs are scored by their interval consonance. Unisons, thirds, fourths, fifths, and sixths are considered consonant; all other intervals are penalized.

```
R_harmony = (1 / (N-1)) · Σ consonance(pitch_i, pitch_{i+1})

where consonance(a, b) = +1.0 if |a-b| mod 12 ∈ {0,3,4,5,7,8,9}
                        = -0.5 otherwise
```

### Rhythm Reward

Duration token ratios between adjacent notes are checked against integer ratios. Rhythmic regularity (ratios close to 1:1, 1:2, etc.) scores higher.

```
R_rhythm = 1 - mean( clip( |round(d_{i+1}/d_i) - d_{i+1}/d_i|, 0, 1 ) )
```

### Perceptual Rewards (Optional)

When audio synthesis is available, two additional reward signals can be enabled:

- **`reward_style`** — Uses a HuggingFace audio classifier to check if the generated audio matches a style prompt (e.g., `"jazz"`, `"classical"`). Score is based on label–prompt token overlap weighted by classifier confidence.
- **`reward_clap`** — Uses [LAION CLAP](https://huggingface.co/laion/clap-htsat-unfused) (zero-shot text–audio classification) to compute a similarity score between the audio and a free-text description. More expressive but slower.

Rewards are combined with configurable weights:

```python
R_total = Σ weight_k · R_k
```

---

## RL Training Loop

The RL trainer (`harmonyrl/training/rl.py`) uses **REINFORCE with an EMA baseline** to reduce variance:

```
∇J(θ) = E[ A · ∇ log π_θ(a | s) ] + α · H[π_θ]

where A = (R - b) / σ_100       # advantage, normalized by rolling std of last 100 rewards
      b = EMA(R, β=0.95)        # running baseline
      H[π_θ]                    # entropy bonus to encourage exploration
```

Each episode:
1. Sample a full token sequence from the current policy (model rollout)
2. Compute reward `R` from the token sequence
3. Backpropagate the policy gradient loss
4. Update the EMA baseline

Mixed-precision training (`torch.cuda.amp`) is used when a GPU is available.

---

## Diffusion Postprocessing

After generating MIDI tokens and synthesizing audio, an optional postprocessing step runs the audio through **AudioLDM2** (`cvssp/audioldm2`). This is a text-conditioned latent diffusion model for audio. The prompt (e.g., `"studio quality jazz trio, warm, clean mix"`) guides the denoising toward a target timbre and style.

This stage is best thought of as an audio-domain polish pass, not a structural one — it won't fix fundamentally incoherent note sequences.

---

## Installation

```bash
git clone https://github.com/SupratikB23/HarmonyRL.git
cd HarmonyRL

python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

## Dataset

This project uses the **[MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)** (MIDI and Audio Edited for Synchronous Tracks and Organization) — approximately 200 hours of virtuosic piano performances across ~1,276 MIDI files, annotated with note-level timing.

Download from [Google Magenta](https://magenta.tensorflow.org/datasets/maestro) and place under `data/maestro/`.

---

## Training

### Step 1 — Supervised Pretraining

```bash
python scripts/train_supervised.py --config configs/supervised_config.yaml
```

Key config options (`configs/supervised_config.yaml`):

```yaml
seed: 42
data:
  root: "data/maestro"
  max_seq_len: 1024
  train_ratio: 0.95
model:
  embed_dim: 512
  hidden: 512
  layers: 3
  dropout: 0.3
train:
  batch_size: 16
  lr: 5e-4
  epochs: 50
  clip_grad_norm: 1.0
  ckpt_dir: "checkpoints"
```

### Step 2 — RL Fine-tuning

```bash
python scripts/train_rl.py --config configs/rl_config.yaml
```

Key config options (`configs/rl_config.yaml`):

```yaml
seed: 123
model:
  embed_dim: 512
  hidden: 768
  layers: 3
  dropout: 0.2
rl:
  episodes: 2000
  rollout_len: 512
  lr: 1e-5
  baseline_beta: 0.95
  entropy_coef: 0.005
```

### Step 3 — Inference

```bash
python scripts/infer.py --ckpt checkpoints/best_model.pt --output_dir outputs/
```

---

## Repository Structure

```
HarmonyRL/
├── configs/
│   ├── supervised_config.yaml
│   └── rl_config.yaml
├── harmonyrl/
│   ├── datasets.py              # MAESTRO MIDI loading & tokenization
│   ├── midi_utils.py            # Tokenization scheme, MIDI ↔ token conversion
│   ├── rewards.py               # Harmony, rhythm, CLAP, style reward functions
│   ├── inference.py             # Sampling + MIDI export
│   ├── postprocess_diffusers.py # AudioLDM2 diffusion postprocessing
│   ├── models/
│   │   ├── lstm.py              # LSTM backbone with weight tying
│   │   └── transformer.py      # Pre-Norm Transformer + RoPE
│   ├── training/
│   │   ├── supervised.py        # Cross-entropy pretraining loop
│   │   └── rl.py                # REINFORCE + EMA baseline RL loop
│   └── utils/
│       ├── evaluation.py        # Standalone harmony reward for evaluation
│       └── logging.py           # Logger setup
├── scripts/
│   ├── train_supervised.py
│   ├── train_rl.py
│   └── infer.py
├── requirements.txt
├── setup.py
└── config.yaml
```

---

## Open Experiments & Ideas

Things actively being explored or worth trying:

- [ ] **Curriculum RL** — start with simple melody rewards, progressively add harmony and structure
- [ ] **PPO instead of REINFORCE** — lower variance updates via clipped surrogate objective
- [ ] **Transformer backbone for RL** — the current RL loop uses LSTM; swap in the Transformer
- [ ] **Multi-instrument extension** — MAESTRO is piano-only; try Lakh MIDI Dataset for ensemble data
- [ ] **Latent diffusion in symbolic space** — postprocess token embeddings rather than raw audio
- [ ] **GAN-based critic** — adversarial reward from a discriminator trained on real MIDI
- [ ] **Larger backbones** — Performer, Music Transformer, or a fine-tuned MusicGen encoder

---

## Tech Stack

| Area | Library |
|---|---|
| Deep Learning | PyTorch ≥ 2.2, torchaudio ≥ 2.2 |
| Symbolic Music | pretty_midi, mido, music21 |
| Audio | librosa, soundfile |
| Datasets / NLP | HuggingFace datasets ≥ 2.20, transformers ≥ 4.41 |
| Diffusion | diffusers ≥ 0.30, accelerate ≥ 0.33 |
| Utilities | numpy, scipy, tqdm, pyyaml |

---

## Acknowledgements

- [Magenta Project](https://magenta.tensorflow.org/) for the MAESTRO dataset
- [LAION CLAP](https://github.com/LAION-AI/CLAP) for text–audio similarity
- [HuggingFace](https://huggingface.co/) for Transformers, Diffusers, and AudioLDM2
- The REINFORCE and PPO literature for RL in sequence generation
