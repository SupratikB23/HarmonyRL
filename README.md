# HarmonyRL 🎶

**Symbolic music generation with supervised pretraining + PPO fine-tuning.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

> **Status:** Active experiment. The v0.2 rewrite replaced the tokenizer, the RL algorithm,
> and the reward — see [What changed in v0.2](#what-changed-in-v02).

---

## What Is This?

HarmonyRL treats music generation as a sequential decision-making problem: pretrain a
language model on real piano performances, then fine-tune it with RL toward outputs that
score well on musical criteria — without letting it drift into degenerate output that
games the reward.

```
MAESTRO MIDI
     │
     ▼  REMI-style event tokens (bar / position / pitch / velocity / duration)
┌─────────────────────┐
│ Supervised pretrain │  cross-entropy, Transformer or LSTM backbone
└──────────┬──────────┘
           │  frozen copy kept as the reference policy
           ▼
┌─────────────────────┐
│  PPO fine-tuning    │  token-level GAE + KL penalty to the reference
│                     │  reward: harmony, key, rhythm, diversity, density
└──────────┬──────────┘
           ▼
      MIDI + audio
```

---

## Tokenization

MIDI becomes a REMI-style event stream. Every id in the vocabulary is reachable — there is
no reserved dead space.

| Token range | Meaning |
|---|---|
| `0–3` | `PAD`, `BOS`, `EOS`, `BAR` |
| `4–19` | Position within the bar (16 steps, 16th-note grid) |
| `20–107` | Pitch (MIDI 21–108, piano range) |
| `108–139` | Velocity (32 bins) |
| `140–171` | Duration (1–32 grid steps) |

Vocabulary size: **172**. A note is four tokens — `Position, Pitch, Velocity, Duration` —
and notes sharing a position stay simultaneous, so **polyphony survives the round trip**.
Timing comes from each file's own tempo map via `pretty_midi.time_to_tick`.

Round-tripping a 1376-note MAESTRO performance returns 1376 notes.

---

## Models

Both backbones share an interface (`forward`, `features`, `sample`) and are selected by
`model.arch` in the config.

### Transformer (default) — `harmonyrl/models/transformer.py`

Pre-norm decoder-only Transformer with rotary position embeddings applied to queries and
keys inside each attention head, causal masking via `scaled_dot_product_attention`, and a
KV cache for generation.

### LSTM — `harmonyrl/models/lstm.py`

Stacked LSTM with LayerNorm and tied input/output embeddings. A projection back to
`embed_dim` keeps tying valid when `hidden != embed_dim`.

Both sample in batches with nucleus (top-p) filtering and pad after `EOS`.

---

## Reward Design

Five symbolic components, all computed on tokens — fast enough for an inner RL loop:

| Component | What it measures |
|---|---|
| `harmony` | Interval consonance between consecutive pitches. **Unison scores 0, not 1** — otherwise "repeat one note forever" is the global optimum. |
| `scale` | Fraction of notes fitting the best-matching major scale. |
| `rhythm` | Simple metric ratios between adjacent durations, scaled by duration variety. |
| `diversity` | Distinct pitch 4-gram ratio. Collapses toward 0 on looped output. |
| `density` | Notes per bar inside a musical range; 0 for an empty sequence. |

```
R_total = Σ weight_k · R_k
```

Sanity check on synthetic sequences, using the weights in `configs/rl_config.yaml`:

| Sequence | harmony | scale | rhythm | diversity | **total** |
|---|---|---|---|---|---|
| One note repeated | 0.00 | 1.00 | 0.00 | 0.03 | **0.26** |
| Uniform-random pitches | 0.26 | 0.70 | 1.00 | 1.00 | **0.67** |
| In-key, varied rhythm | 0.51 | 1.00 | 0.97 | 1.00 | **0.80** |

The degenerate policy scores worst. `tests/test_rewards.py` asserts this.

---

## RL Training Loop

`harmonyrl/training/rl.py` runs **PPO** over batched rollouts:

```
r_t   = -β · (log π(a_t) − log π_ref(a_t))      per-token KL penalty
r_T  += R(sequence)                              sequence reward at the final token
A_t   = GAE(r, V, γ, λ)                          token-level credit assignment
L     = −min(ρ_t A_t, clip(ρ_t, 1±ε) A_t) + c_v·(V_t − G_t)² − c_H·H[π]
```

- A frozen copy of the supervised model is the **reference policy**. The KL term is what
  stops the policy collapsing onto whatever degenerate output maximizes the reward.
- A value head over the backbone's hidden states supplies per-token baselines, so credit
  goes to the tokens responsible instead of being averaged across the whole episode.
- Advantages are whitened over the valid (non-padding) tokens of each batch.
- Dropout is disabled during RL so the importance ratio starts at exactly 1.

---

## Evaluation

`harmonyrl/utils/evaluation.py` reports held-out perplexity plus per-sample metrics:
`distinct_4`, `max_repeat_run`, `pitch_class_entropy`, and every reward component.

`max_repeat_run` is the direct read-out for the degenerate-loop failure mode — watch it
alongside reward during RL, because reward rising while it climbs means reward hacking.

---

## Installation

```bash
git clone https://github.com/SupratikB23/HarmonyRL.git
cd HarmonyRL

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .              # optional; the scripts also run without it
```

The audio extras (`transformers`, `diffusers`, `accelerate`) are only needed for the
optional AudioLDM2 pass: `pip install -e ".[audio]"`.

---

## Dataset

[MAESTRO](https://magenta.tensorflow.org/datasets/maestro) — roughly 200 hours of virtuosic
piano performance across ~1276 MIDI files. Download it and place the files under
`data/maestro/`.

The corpus is tokenized once and cached under `.cache/` (~9 min for all 1276 files, 28.5M
tokens); training then reads fixed-length chunks with a 50% stride, so one long performance
yields many samples rather than a single truncated sequence — 53,594 training chunks at
`max_seq_len` 1024, against 1276 truncated samples under v0.1. Train/validation splits are made **by recording group**, so movements
of the same performance cannot straddle the split.

---

## Training

### Step 1 — Supervised pretraining

```bash
python scripts/train_supervised.py --config configs/supervised_config.yaml
```

Saves `checkpoints/<arch>_supervised.pt`. Checkpoints carry their own model config, so
inference and RL rebuild the exact architecture — no hardcoded dimensions.

### Step 2 — PPO fine-tuning

```bash
python scripts/train_rl.py --config configs/rl_config.yaml
```

Requires the supervised checkpoint named by `train.init_from`.

### Step 3 — Inference

```bash
python scripts/infer.py --n_samples 4 --output_dir outputs/
```

Writes MIDI (and WAV, unless `--no_audio`) and logs the evaluation metrics per sample.

Training needs a GPU. See [`notebooks/`](notebooks/) for ready-to-run molab and
Kaggle notebooks that fetch MAESTRO, train both stages, and generate samples.

### Tests

```bash
pytest
```

Covers vocabulary density, round-trip polyphony, decoder tolerance to malformed output,
reward non-hackability, transformer causality, KV-cache equivalence, and checkpoint
round-trips.

---

## Repository Structure

```
HarmonyRL/
├── configs/
│   ├── supervised_config.yaml
│   └── rl_config.yaml
├── harmonyrl/
│   ├── datasets.py              # cached chunking, group-wise splits
│   ├── midi_utils.py            # REMI-style tokenizer, MIDI <-> tokens
│   ├── rewards.py               # harmony, scale, rhythm, diversity, density
│   ├── inference.py             # sampling -> MIDI -> audio
│   ├── postprocess_diffusers.py # optional AudioLDM2 timbre pass
│   ├── models/
│   │   ├── lstm.py
│   │   ├── transformer.py       # RoPE + causal SDPA + KV cache
│   │   └── sampling.py          # nucleus filtering
│   ├── training/
│   │   ├── supervised.py        # warmup + cosine, AMP, early stopping
│   │   └── rl.py                # PPO + GAE + KL-to-reference
│   └── utils/
│       ├── checkpoint.py        # self-describing checkpoints
│       ├── evaluation.py        # perplexity + generation metrics
│       └── logging.py
├── notebooks/                   # Kaggle / molab GPU training notebooks
│   ├── harmonyrl_molab.py       # marimo notebook (molab)
│   ├── harmonyrl_kaggle.ipynb
│   └── configs/                 # GPU-sized training presets
├── scripts/                     # train_supervised / train_rl / infer
├── tests/
└── pyproject.toml
```

---

## What changed in v0.2

The v0.1 pipeline had five compounding problems, all addressed here:

1. **The tokenizer discarded polyphony.** Notes were laid end to end, so a 1376-note
   performance decoded to 311 sequential monophonic notes. Replaced with the REMI-style
   scheme above.
2. **62% of the vocabulary was unreachable.** `VOCAB_SIZE` was 261 with only 99 legal ids.
   Now 172 ids, all reachable, with a test that enforces it.
3. **One truncated sample per file.** 1276 files gave 1276 samples, each covering a few
   seconds of music. Now chunked with stride across the whole corpus.
4. **The RL loop could not complete an episode** — `GradScaler.step` was called without
   `scale(...).backward()`, required config keys were missing, and the entire episode
   shared one averaged log-probability, so there was no credit assignment. Replaced with
   PPO.
5. **The reward was trivially hackable.** Unison counted as consonant, so a stuck note
   scored a perfect 1.0. Fixed, and backed by a diversity term plus a KL anchor to the
   reference policy.

Also fixed: the Transformer had **no causal mask** and attended to future tokens; RoPE was
applied once to the embedding rather than to queries and keys; top-p filtering was off by
one; weight tying broke whenever `hidden != embed_dim`; gradients were clipped before
being unscaled; inference hardcoded dimensions matching no config; and the
train/validation split leaked movements of the same recording.

Checkpoints from v0.1 are not loadable — the vocabulary changed, so they are rejected with
an explicit error rather than silently mismatching. Retrain from scratch.

---

## Open Experiments & Ideas

- [ ] **GRPO** alongside PPO — group-relative advantages, no critic; compare on one reward
- [ ] **Learned reward model** — a discriminator on real vs. generated MIDI instead of hand-written rules
- [ ] **Bar-level rewards** rather than one sequence-level scalar
- [ ] **Multi-instrument** — Lakh MIDI for ensemble data
- [ ] **Longer context** — sliding-window attention beyond 2048 tokens
- [ ] **Human A/B listening harness**, to check that the symbolic metrics track perceived quality

---

## Tech Stack

| Area | Library |
|---|---|
| Deep Learning | PyTorch ≥ 2.2 |
| Symbolic Music | pretty_midi |
| Audio | soundfile |
| Diffusion (optional) | diffusers, transformers, accelerate |
| Utilities | numpy, tqdm, pyyaml, pytest |

---

## Acknowledgements

- [Magenta](https://magenta.tensorflow.org/) for the MAESTRO dataset
- The REMI tokenization scheme (Huang & Yang, 2020)
- PPO (Schulman et al., 2017) and the RLHF KL-anchoring recipe
