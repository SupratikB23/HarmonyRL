# Training notebooks

Training HarmonyRL needs a GPU. These notebooks fetch the code and MAESTRO, run both
training stages, and generate samples — nothing to upload.

| | [molab](https://molab.marimo.io) | [Kaggle](https://kaggle.com/code) |
|---|---|---|
| GPU | RTX Pro 6000 Blackwell, 96 GB VRAM | T4 (16 GB) or P100 |
| Session cap | 12 h | 12 h, ~30 GPU-h/week |
| Notebook | `harmonyrl_molab.py` | `harmonyrl_kaggle.ipynb` |
| Format | marimo (`.py`) | Jupyter (`.ipynb`) |
| Cost | free | free |

**Use molab.** The Blackwell card is far ahead of a T4, and you can raise `batch_size`
well past the preset without running out of memory.

---

## What you feed the platform

Just the notebook. Everything else is fetched at runtime:

- **Code** — cloned from your GitHub repo. Push your work first, then set `BRANCH` in the
  notebook's cell 1 to whatever branch you pushed.
- **Data** — MAESTRO v3.0.0 MIDI (58 MB, 1276 files) downloaded straight from Magenta.
  You do **not** need to upload your local `data/maestro/`.
- **Configs** — `configs/supervised_gpu.yaml` and `configs/rl_gpu.yaml` in this folder,
  which ride along with the repo clone.

---

## molab

1. Go to [molab.marimo.io](https://molab.marimo.io) and create a notebook.
2. Upload `harmonyrl_molab.py` through the sidebar file manager, or open it from GitHub.
3. **Click the notebook specs button in the header and attach the GPU.** The notebook
   raises an error if you skip this — a CPU run would take weeks.
4. Run the cells top to bottom.

molab only persists files you uploaded through the sidebar or cached with
`mo.persistent_cache`. **Download `checkpoints/*.pt` before the session ends**, or use the
optional Hugging Face upload cell at the bottom.

## Kaggle

1. New Notebook → **File → Import Notebook** → upload `harmonyrl_kaggle.ipynb`.
2. In the right-hand panel set **Accelerator → GPU T4 x2** and **Internet → On**.
   Both are required; the notebook checks the first and fails fast without it.
3. Run all.

`/kaggle/working` is wiped when the session ends. Save checkpoints from the **Output** tab,
or write them out as a Kaggle Dataset so a later session can mount them as an input.

---

## The GPU presets

`configs/supervised_gpu.yaml` is a ~25M-param Transformer (`d_model` 512, 8 layers) over
sequences of 1024 tokens at batch 32.

Measured on the full 1276-file corpus:

| | |
|---|---|
| Tokens cached | 28.5M |
| Train / val chunks | 53,594 / 2,441 |
| Steps per epoch @ batch 32 | 1,674 |
| One-time tokenizing pass | ~9 min |

30 epochs is ~50k steps. Wall-clock depends on the card — expect a couple of hours on
molab's Blackwell, appreciably longer on a T4. Early stopping usually trips first.

**On a 16 GB card**, if you hit OOM: drop `batch_size` to 8 and `max_seq_len` to 512.
**On molab**, you can raise `batch_size` to 64+ and leave everything else alone.

---

## Reading the training output

**Stage 1 — supervised.** Watch `val ppl`. It should fall well under 10. A random model
sits near 172 (the vocabulary size), so anything close to that means it is not learning.

**Stage 2 — PPO.** Watch `R` and `diversity` *together*:

| What you see | What it means |
|---|---|
| `R` up, `diversity` ~1.0 | working |
| `R` up, `diversity` falling | **reward hacking** — stop, raise `kl_coef` or the `diversity` weight |
| `kl` drifting up slowly | normal |
| `kl` jumping | the policy is running away from the reference; lower `lr` |

**Stage 3 — samples.** `max_repeat_run` in low single digits is healthy. A large value
means the model is stuck on one note no matter what the reward reports.
