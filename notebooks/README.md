# Training notebooks

Training HarmonyRL needs a GPU. These notebooks fetch the code and MAESTRO, run both
stages, and generate samples. Nothing to upload but the notebook itself.

| | [molab](https://molab.marimo.io) | [Kaggle](https://kaggle.com/code) |
|---|---|---|
| GPU | RTX Pro 6000 Blackwell, 96 GB | T4 (16 GB) or P100 |
| Session cap | 12 h | 12 h, about 30 GPU-h per week |
| Notebook | `harmonyrl_molab.py` | `harmonyrl_kaggle.ipynb` |
| Format | marimo (.py) | Jupyter (.ipynb) |

Prefer molab. The Blackwell card is far ahead of a T4, and you can raise `batch_size` well
past the preset without running out of memory.

## What gets fetched at runtime

- Code, cloned from GitHub. Push your work first, then set `BRANCH` in the clone cell.
- MAESTRO v3.0.0 MIDI, 58 MB, 1276 files, pulled straight from Magenta. Your local
  `data/maestro/` is not needed.
- Configs, from `configs/` in this folder, which ride along with the clone.

## molab

1. Create a notebook at [molab.marimo.io](https://molab.marimo.io) and upload
   `harmonyrl_molab.py`.
2. Click the notebook specs button in the header and attach the GPU. The notebook stops
   with an error if you skip this.
3. Run the cells from the top. They stop and wait at a Start button before anything long
   runs.

marimo schedules cells by data dependency, not page order, so each stage consumes the
previous stage's result and the sequence is enforced for you. The two multi-hour stages
sit behind run buttons, so they never start on load and never re-run themselves.

molab only persists files uploaded through the sidebar or cached with
`mo.persistent_cache`. Download `checkpoints/*.pt` before the session ends, or use the
Hugging Face upload cell at the bottom.

## Kaggle

1. File, Import Notebook, upload `harmonyrl_kaggle.ipynb`.
2. In the right-hand panel set Accelerator to GPU T4 x2 and Internet to On. Both are
   required.
3. Run all.

`/kaggle/working` is wiped when the session ends. Save checkpoints from the Output tab, or
write them out as a Kaggle Dataset so a later session can mount them as an input.

## The GPU presets

`configs/supervised_gpu.yaml` is a 25.3M-parameter Transformer, `d_model` 512 over 8
layers, sequences of 1024 tokens at batch 32.

Measured on the full corpus:

| | |
|---|---|
| Tokens cached | 28.5M |
| Train / val chunks | 53,594 / 2,441 |
| Steps per epoch at batch 32 | 1,674 |
| One-time tokenizing pass | about 9 min |
| Checkpoint size | 101 MB |

30 epochs is roughly 50k steps; early stopping usually trips first. A reference run stopped
at epoch 12 with validation perplexity 3.16.

On a 16 GB card, if you hit OOM, drop `batch_size` to 8 and `max_seq_len` to 512. On molab
you can raise `batch_size` to 64 or beyond and leave everything else alone.

## Reading the output

**Stage 1, supervised.** Watch `val ppl`. It should fall steadily and land well under 10.
A random model sits near 172, the vocabulary size. After the tokenizing bar finishes, the
only output is one line per epoch, so a quiet cell is normal rather than a hang.

**Stage 2, PPO.** Watch `R` and `diversity` together.

| Reading | Meaning |
|---|---|
| `R` up, `diversity` near 1.0 | working |
| `R` up, `diversity` falling | reward hacking; stop and raise `kl_coef` or the `diversity` weight |
| `kl` drifting up slowly | normal |
| `kl` jumping | policy running from the reference; lower `lr` |

**Stage 3, samples.** `max_repeat_run` in low single digits is healthy. A large value means
the model is stuck on one note regardless of what the reward reports.
