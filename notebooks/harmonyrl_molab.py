import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # HarmonyRL — training run

        Supervised pretraining on MAESTRO, then PPO fine-tuning.

        **Before running:** click the notebook specs button in the header and attach a GPU.
        The cell below refuses to continue without one — a CPU run would take weeks.

        Sessions end after 12 hours. Stage 1 takes roughly 4-6 hours, so run it, download
        the checkpoint from the last cell, and start stage 2 in a fresh session if needed.
        """
    )
    return (mo,)


@app.cell
def _():
    import subprocess
    import sys
    from pathlib import Path

    def sh(cmd, cwd=None):
        """Run a shell command, streaming output into the cell."""
        proc = subprocess.run(cmd, shell=True, cwd=cwd, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"failed ({proc.returncode}): {cmd}")
        return proc.stdout

    return Path, sh, subprocess, sys


@app.cell
def _(sys):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU attached. Click the notebook specs button in the header, "
            "enable the GPU, and re-run. Training on CPU is not viable here."
        )

    print("torch     ", torch.__version__)
    print("gpu       ", torch.cuda.get_device_name(0))
    print("vram (GB) ", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print("python    ", sys.version.split()[0])
    return (torch,)


@app.cell
def _(mo):
    mo.md("""## 1. Get the code""")
    return


@app.cell
def _(Path, sh):
    REPO_URL = "https://github.com/SupratikB23/HarmonyRL.git"
    BRANCH = "main"  # change if you pushed the rewrite to a branch
    REPO = Path("HarmonyRL")

    if not REPO.exists():
        sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO}")
    else:
        sh("git pull", cwd=REPO)

    sh("pip install -q -r requirements.txt", cwd=REPO)
    print("\nrepo ready at", REPO.resolve())
    return BRANCH, REPO, REPO_URL


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Get MAESTRO

        Pulled straight from Magenta — nothing to upload. ~58 MB zip, 1276 MIDI files.
        """
    )
    return


@app.cell
def _(REPO, sh):
    MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    DATA = REPO / "data" / "maestro"

    if not DATA.exists() or not any(DATA.rglob("*.mid*")):
        DATA.mkdir(parents=True, exist_ok=True)
        sh(f"curl -L -o maestro.zip {MAESTRO_URL}", cwd=REPO)
        sh(f"unzip -q -o maestro.zip -d {DATA}", cwd=REPO)
        sh("rm -f maestro.zip", cwd=REPO)

    n_midi = len(list(DATA.rglob("*.mid*")))
    print(f"{n_midi} MIDI files under {DATA}")
    assert n_midi > 1000, "expected ~1276 files; check the download"
    return DATA, MAESTRO_URL, n_midi


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Smoke test

        Runs the test suite before spending GPU hours. If this fails, stop and fix it —
        a broken tokenizer wastes the whole run.
        """
    )
    return


@app.cell
def _(REPO, sh):
    sh("python -m pytest tests -q", cwd=REPO)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Supervised pretraining

        ~25M params over ~41M tokens. Expect 4-6 hours for the full 30 epochs; early
        stopping usually trips before that. Watch **val ppl** — it should fall well under 10.

        The first run tokenizes all 1276 files and caches them to `.cache/`, which takes a
        few minutes. Later runs reuse it.
        """
    )
    return


@app.cell
def _(REPO, sh):
    sh("python scripts/train_supervised.py --config notebooks/configs/supervised_gpu.yaml",
       cwd=REPO)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. PPO fine-tuning

        Starts from the supervised checkpoint and keeps a frozen copy of it as the
        reference policy.

        **Watch two numbers together:**

        - `R` rising is only good if `diversity` stays near 1.0.
        - If `R` rises while `diversity` falls, the policy is hacking the reward — kill it
          and raise `kl_coef`, or raise the `diversity` weight.
        - `kl` should drift up slowly. A sudden jump means the policy is running away from
          the reference.
        """
    )
    return


@app.cell
def _(REPO, sh):
    sh("python scripts/train_rl.py --config notebooks/configs/rl_gpu.yaml", cwd=REPO)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Generate and inspect

        `max_repeat_run` is the honest check: low single digits is healthy, a large number
        means the model is stuck on a note regardless of what the reward says.
        """
    )
    return


@app.cell
def _(REPO, sh):
    sh("python scripts/infer.py --n_samples 6 --max_new_tokens 1024 "
       "--output_dir outputs --no_audio", cwd=REPO)
    return


@app.cell
def _(REPO):
    import glob

    import pretty_midi

    for path in sorted(glob.glob(str(REPO / "outputs" / "*.mid"))):
        pm = pretty_midi.PrettyMIDI(path)
        notes = pm.instruments[0].notes if pm.instruments else []
        print(f"{path.split('/')[-1]:20s} {len(notes):5d} notes  "
              f"{pm.get_end_time():6.1f}s")
    return glob, path, pm, pretty_midi


@app.cell
def _(mo):
    mo.md(
        """
        ## 7. Take your checkpoints with you

        molab only persists files uploaded through the sidebar or written via
        `mo.persistent_cache`. **Download these before the session ends** — right-click
        them in the sidebar file tree, or push them to a Hugging Face repo.
        """
    )
    return


@app.cell
def _(REPO, sh):
    sh("ls -lh checkpoints outputs", cwd=REPO)
    print("\nDownload checkpoints/*.pt from the sidebar file tree now.")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Optional — push to Hugging Face instead of downloading by hand

        Safer for a 12-hour session. Set `HF_TOKEN` in the notebook's secrets first, then
        uncomment and run.
        """
    )
    return


@app.cell
def _():
    # import os
    # from huggingface_hub import HfApi
    #
    # api = HfApi(token=os.environ["HF_TOKEN"])
    # repo_id = "your-username/harmonyrl"
    # api.create_repo(repo_id, exist_ok=True)
    # api.upload_folder(folder_path="HarmonyRL/checkpoints", repo_id=repo_id,
    #                   path_in_repo="checkpoints")
    return


if __name__ == "__main__":
    app.run()
