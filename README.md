# HarmonyRL

Symbolic piano music generation: supervised pretraining on MAESTRO, then PPO fine-tuning
against musical reward functions.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Hugging%20Face-Supratik23%2FHarmonyRL-yellow?logo=huggingface)](https://huggingface.co/Supratik23/HarmonyRL)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

Trained weights: [Supratik23/HarmonyRL](https://huggingface.co/Supratik23/HarmonyRL)

---

## What it does

A language model is trained on real piano performances, then fine-tuned with RL toward
outputs that score well on musical criteria. The hard part is not raising the reward, it
is raising it without the policy collapsing onto degenerate output that games the metric.
A KL penalty against the frozen pretrained model is what prevents that.

```
MAESTRO MIDI
  -> REMI-style tokens (bar / position / pitch / velocity / duration)
  -> supervised pretraining, cross-entropy            [Transformer or LSTM]
  -> PPO fine-tuning, token-level GAE + KL anchor     [reward: 5 symbolic terms]
  -> MIDI, audio, MP3
```

## Results

Trained on the full 1276-file corpus, 25.3M-parameter Transformer.

| | |
|---|---|
| Validation perplexity | 3.16 (random baseline 172) |
| Stopped at | epoch 12, early stopping |
| PPO | 3000 iterations |
| Tokens / train chunks | 28.5M / 53,594 |

Generated samples hold 11 to 37 simultaneous note onsets, use 17 distinct velocity levels,
and score 0.98 to 1.00 on `distinct_4` with a longest repeated-note run of 3 to 4. No
degenerate loops. Weakest axis is tonality: `scale` sits at 0.65 to 0.74, and pitch spreads
across the full 88-key range, so the output wanders between registers.

---

## Tokenization

MIDI becomes a REMI-style event stream. Every id in the vocabulary is reachable.

| Range | Meaning |
|---|---|
| 0-3 | PAD, BOS, EOS, BAR |
| 4-19 | Position in bar, 16th-note grid |
| 20-107 | Pitch, MIDI 21-108 |
| 108-139 | Velocity, 32 bins |
| 140-171 | Duration, 1-32 grid steps |

Vocabulary size 172. A note is four tokens: Position, Pitch, Velocity, Duration. Notes
sharing a position stay simultaneous, so polyphony survives the round trip. Timing comes
from each file's own tempo map. Round-tripping a 1376-note performance returns 1376 notes
with a byte-identical token stream.

## Models

Selected by `model.arch` in the config; both expose `forward`, `features`, `sample`.

**Transformer** (default) is a pre-norm decoder-only stack with rotary embeddings applied
to queries and keys inside each attention head, causal masking via
`scaled_dot_product_attention`, and a KV cache for generation.

**LSTM** is a stacked LSTM with LayerNorm and tied input/output embeddings, with a
projection that keeps tying valid when `hidden` differs from `embed_dim`.

Both sample in batches with nucleus filtering, and never emit PAD or BOS mid-stream.

## Reward design

Five symbolic terms, all computed on tokens so they are cheap enough for an inner RL loop.

| Term | Measures |
|---|---|
| harmony | Interval consonance between consecutive pitches. Unison scores 0, not 1, or "repeat one note forever" becomes the global optimum. |
| scale | Fraction of notes fitting the best-matching major scale. |
| rhythm | Simple metric ratios between adjacent durations, scaled by duration variety. |
| diversity | Distinct pitch 4-gram ratio. Collapses toward 0 on looped output. |
| density | Notes per bar inside a musical range; 0 for an empty sequence. |

Checked against synthetic sequences, using the weights in `configs/rl_config.yaml`:

| Sequence | harmony | scale | rhythm | diversity | total |
|---|---|---|---|---|---|
| One note repeated | 0.00 | 1.00 | 0.00 | 0.03 | 0.26 |
| Uniform-random pitches | 0.26 | 0.70 | 1.00 | 1.00 | 0.67 |
| In-key, varied rhythm | 0.51 | 1.00 | 0.97 | 1.00 | 0.80 |

The degenerate policy scores worst. `tests/test_rewards.py` asserts this.

## RL loop

```
r_t   = -beta * (log pi(a_t) - log pi_ref(a_t))     per-token KL penalty
r_T  += R(sequence)                                  sequence reward at final live token
A_t   = GAE(r, V, gamma, lambda)                     token-level credit assignment
L     = -min(rho_t A_t, clip(rho_t) A_t) + c_v * value loss - c_H * entropy
```

A frozen copy of the pretrained model is the reference policy. A value head over the
backbone's hidden states supplies per-token baselines, so credit reaches the tokens
responsible rather than being averaged across the episode. Advantages are whitened over
live tokens. Dropout is off during RL so the importance ratio starts at exactly 1.

## Evaluation

`harmonyrl/utils/evaluation.py` reports held-out perplexity plus per-sample `distinct_4`,
`max_repeat_run`, `pitch_class_entropy` and every reward term. `max_repeat_run` is the
direct read-out for the degenerate-loop failure mode: reward rising while it climbs means
reward hacking.

---

## Install

```bash
git clone https://github.com/SupratikB23/HarmonyRL.git
cd HarmonyRL
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional extras: `pip install -e ".[audio]"` for the AudioLDM2 pass, `pip install lameenc`
for MP3 export.

## Dataset

[MAESTRO](https://magenta.tensorflow.org/datasets/maestro), about 200 hours of piano
performance across 1276 MIDI files, placed under `data/maestro/`.

The corpus is tokenized once into `.cache/`, roughly 9 minutes for 28.5M tokens. Training
reads fixed-length chunks at 50 percent stride, giving 53,594 training chunks at
`max_seq_len` 1024. Train and validation are split by recording group, so movements of one
performance cannot straddle the split.

## Training

Training needs a GPU. See [`notebooks/`](notebooks/) for ready-to-run molab and Kaggle
notebooks that fetch MAESTRO, run both stages, and generate samples.

```bash
python scripts/train_supervised.py --config configs/supervised_config.yaml
python scripts/train_rl.py         --config configs/rl_config.yaml
python scripts/infer.py --n_samples 4 --output_dir outputs/
```

Checkpoints carry their own model config, so inference and RL rebuild the exact
architecture with no hardcoded dimensions.

### MP3 export

```bash
pip install lameenc
python scripts/to_mp3.py                        # outputs/*.mid -> outputs/mp3/*.mp3
python scripts/to_mp3.py --soundfont piano.sf2  # real samples, needs FluidSynth
```

Standalone and independent of the training pipeline. `lameenc` ships LAME as a wheel, so
no ffmpeg install is required. Without a soundfont it renders with a built-in additive
piano: inharmonic partials with per-partial decay, velocity-dependent brightness, damper
on note-off, stereo spread by pitch, a small convolution room, and percentile
normalization with a soft limiter.

| Flag | Default | |
|---|---|---|
| `--brightness` | 1400 | tone rolloff in Hz, raise for a brighter piano |
| `--max_note` | off | cap note length in seconds, thins long-note pileups |
| `--reverb` | 0.22 | 0 disables |
| `--soundfont` | off | .sf2 path, needs FluidSynth |

Also `--bitrate`, `--sr`, `--input_dir`, `--output_dir`, `--overwrite`.

### Tests

```bash
pytest
```

46 tests covering vocabulary density, round-trip polyphony and velocity, decoder tolerance
to malformed output, reward non-hackability, transformer causality, KV-cache equivalence,
dataset chunking and split isolation, PPO masking and GAE, and checkpoint round-trips.

---

## Layout

```
configs/                     supervised_config.yaml, rl_config.yaml
harmonyrl/
  midi_utils.py              REMI tokenizer, MIDI to tokens and back
  datasets.py                cached chunking, group-wise splits
  rewards.py                 harmony, scale, rhythm, diversity, density
  inference.py               sampling to MIDI to audio
  postprocess_diffusers.py   optional AudioLDM2 timbre pass
  models/                    transformer.py, lstm.py, sampling.py
  training/                  supervised.py, rl.py
  utils/                     checkpoint.py, evaluation.py, logging.py
notebooks/                   molab and Kaggle GPU notebooks, GPU configs
scripts/                     train_supervised, train_rl, infer, to_mp3
tests/
```

## What changed in v0.2

The v0.1 pipeline had five compounding problems:

1. The tokenizer discarded polyphony. Notes were laid end to end, so a 1376-note
   performance decoded to 311 sequential monophonic notes.
2. 62 percent of the vocabulary was unreachable: 261 ids with only 99 legal.
3. One truncated sample per file, so 1276 files gave 1276 samples of a few seconds each.
4. The RL loop could not complete an episode. `GradScaler.step` was called without
   `scale(...).backward()`, config keys were missing, and one averaged log-probability was
   shared across the episode, so there was no credit assignment.
5. The reward was trivially hackable. Unison counted as consonant, so a stuck note scored
   a perfect 1.0.

Also fixed: the Transformer had no causal mask and attended to future tokens; rotary
embeddings were applied once to the embedding rather than to queries and keys; the cached
decode path dropped masking for multi-token prefills; top-p filtering was off by one;
weight tying broke when `hidden` differed from `embed_dim`; the LSTM never initialized its
tied embedding, starting at loss 41 against a uniform baseline of 5.15; velocity decoding
was off by one bin, shifting every note louder on each round trip; gradients were clipped
before unscaling; validation used the label-smoothed loss, biasing early stopping;
inference hardcoded dimensions matching no config; and the split leaked movements of the
same recording.

v0.1 checkpoints are not loadable. The vocabulary changed, so they are rejected with an
explicit error rather than silently mismatching.

## Roadmap

- GRPO alongside PPO: group-relative advantages, no critic, compared on one reward
- A learned reward model, a discriminator on real against generated MIDI
- Bar-level rewards instead of one sequence-level scalar
- Register control, to stop pitch wandering across the full 88 keys
- Multi-instrument, using Lakh MIDI
- Human A/B listening tests, to check the symbolic metrics track perceived quality

## Acknowledgements

MAESTRO from [Magenta](https://magenta.tensorflow.org/). REMI tokenization from Huang and
Yang, 2020. PPO from Schulman et al., 2017, with the KL-anchoring recipe from RLHF.

Licensed under Apache 2.0.
