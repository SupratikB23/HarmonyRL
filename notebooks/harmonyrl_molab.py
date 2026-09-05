import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # HarmonyRL — training run

        Supervised pretraining on MAESTRO, then PPO fine-tuning.

        **Attach a GPU first:** click the notebook specs button in the header. The check
        below refuses to continue without one — a CPU run would take weeks.

        marimo runs cells by *data dependency*, not by position on the page. Each stage
        below consumes the previous stage's result, so they are forced into the right
        order: data → tests → pretraining → PPO → samples.

        The two long stages sit behind **Start** buttons so they never run on their own,
        and never re-run themselves.
        """
    )
    return


@app.cell
def _():
    import subprocess
    import sys
    from pathlib import Path

    def sh(cmd, cwd=None):
        """Run a command, streaming its output into the cell as it arrives."""
        proc = subprocess.Popen(
            cmd, shell=True, cwd=cwd, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        out = []
        for line in proc.stdout:
            print(line, end="")
            out.append(line)
        proc.wait()
        if proc.returncode:
            raise RuntimeError(f"failed ({proc.returncode}): {cmd}")
        return "".join(out)

    return Path, sh, subprocess, sys


@app.cell
def _(sys):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU attached. Click the notebook specs button in the header, "
            "enable the GPU, then re-run this cell."
        )

    print("torch     ", torch.__version__)
    print("gpu       ", torch.cuda.get_device_name(0))
    print("vram (GB) ", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print("python    ", sys.version.split()[0])
    gpu_ok = True
    return gpu_ok, torch


@app.cell
def _(mo):
    mo.md("""## 1. Get the code""")
    return


@app.cell
def _(Path, gpu_ok, sh):
    assert gpu_ok
    REPO_URL = "https://github.com/SupratikB23/HarmonyRL.git"
    BRANCH = "main"
    REPO = Path("HarmonyRL").resolve()

    if not REPO.exists():
        sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO}")
    else:
        sh("git pull", cwd=REPO)

    sh("pip install -q -r requirements.txt", cwd=REPO)
    print("\nrepo ready at", REPO)
    return BRANCH, REPO, REPO_URL


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Get MAESTRO

        Downloaded straight from Magenta — nothing to upload. ~58 MB, 1276 MIDI files.

        Uses `urllib` + `zipfile` rather than `curl`/`unzip`, so it does not depend on
        those being installed in the container.
        """
    )
    return


@app.cell
def _(REPO):
    import urllib.request
    import zipfile

    MAESTRO_URL = (
        "https://storage.googleapis.com/magentadata/datasets/"
        "maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    )
    DATA = REPO / "data" / "maestro"
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = REPO / "maestro.zip"

    if not any(DATA.rglob("*.mid*")):
        print("downloading MAESTRO ...")
        urllib.request.urlretrieve(MAESTRO_URL, zip_path)
        print(f"  {zip_path.stat().st_size / 1e6:.1f} MB, extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA)
        zip_path.unlink()

    n_midi = len(list(DATA.rglob("*.mid*")))
    print(f"\n{n_midi} MIDI files under {DATA}")
    if n_midi < 1000:
        raise RuntimeError(
            f"Expected ~1276 MIDI files, found {n_midi}. The download or extract failed; "
            f"delete {DATA} and re-run this cell."
        )
    return DATA, MAESTRO_URL, n_midi, urllib, zip_path, zipfile


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Smoke test

        Runs the suite before spending GPU hours. If this fails, stop — a broken
        tokenizer would waste the whole run.
        """
    )
    return


@app.cell
def _(REPO, n_midi, sh):
    assert n_midi >= 1000
    sh("python -m pytest tests -q", cwd=REPO)
    tests_ok = True
    return (tests_ok,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Supervised pretraining

        ~25M params over 28.5M tokens: 53,594 chunks, 1,674 steps per epoch at batch 32.
        30 epochs is ~50k steps. Early stopping usually trips first.

        The first run tokenizes all 1276 files into `.cache/` — about 9 minutes, with a
        progress bar. Later runs reuse the cache. After that the only output is one
        line per epoch, so a quiet cell is normal, not a hang.

        Watch **`val ppl`** — it should fall steadily and land well under 10. A random
        model sits near 172 (the vocabulary size).

        Press **Start pretraining** when you are ready. Nothing happens until you do.
        """
    )
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start pretraining")
    train_button
    return (train_button,)


@app.cell
def _(REPO, mo, sh, tests_ok, train_button):
    mo.stop(not train_button.value, mo.md("*Waiting for **Start pretraining**.*"))
    assert tests_ok

    sh("python -u scripts/train_supervised.py "
       "--config notebooks/configs/supervised_gpu.yaml", cwd=REPO)

    sup_ckpt = REPO / "checkpoints" / "transformer_supervised.pt"
    if not sup_ckpt.exists():
        raise RuntimeError(f"training finished but {sup_ckpt} is missing")
    print(f"\n{sup_ckpt.name}: {sup_ckpt.stat().st_size / 1e6:.1f} MB")
    sup_done = True
    return sup_ckpt, sup_done


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Download the checkpoint now

        molab only persists files uploaded through the sidebar or cached with
        `mo.persistent_cache`. **Open the sidebar file tree and download
        `HarmonyRL/checkpoints/transformer_supervised.pt` before doing anything else.**

        If the session ends without it, the whole pretraining run is gone.
        """
    )
    return


@app.cell
def _(REPO, mo, sh, sup_done):
    mo.stop(not sup_done)
    sh("ls -lh checkpoints", cwd=REPO)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. PPO fine-tuning

        Starts from the supervised checkpoint and keeps a frozen copy of it as the
        reference policy.

        **Watch `R` and `diversity` together:**

        | What you see | What it means |
        |---|---|
        | `R` up, `diversity` ≈ 1.0 | working |
        | `R` up, `diversity` falling | **reward hacking** — stop, raise `kl_coef` |
        | `kl` drifting up slowly | normal |
        | `kl` jumping | policy running from the reference; lower `lr` |
        """
    )
    return


@app.cell
def _(mo):
    rl_button = mo.ui.run_button(label="Start PPO fine-tuning")
    rl_button
    return (rl_button,)


@app.cell
def _(REPO, mo, rl_button, sh, sup_done):
    mo.stop(not rl_button.value, mo.md("*Waiting for **Start PPO fine-tuning**.*"))
    assert sup_done

    sh("python -u scripts/train_rl.py --config notebooks/configs/rl_gpu.yaml", cwd=REPO)
    rl_done = True
    return (rl_done,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 7. Generate and inspect

        `max_repeat_run` is the honest check: low single digits is healthy. A large value
        means the model is stuck on one note regardless of what the reward reports.
        """
    )
    return


@app.cell
def _(REPO, mo, rl_done, sh):
    mo.stop(not rl_done)
    sh("python -u scripts/infer.py --n_samples 6 --max_new_tokens 1024 "
       "--output_dir outputs --no_audio", cwd=REPO)
    infer_done = True
    return (infer_done,)


@app.cell
def _(REPO, infer_done, mo):
    mo.stop(not infer_done)
    import pretty_midi

    midi_files = sorted((REPO / "outputs").glob("*.mid"))
    if not midi_files:
        print("no MIDI written -- check the inference cell above")
    for midi_path in midi_files:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = pm.instruments[0].notes if pm.instruments else []
        print(f"{midi_path.name:20s} {len(notes):5d} notes  {pm.get_end_time():6.1f}s")
    return (midi_files,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 8. Take everything with you

        Download `checkpoints/*.pt` and `outputs/*.mid` from the sidebar file tree, or
        push them to Hugging Face with the cell below (set `HF_TOKEN` in the notebook's
        secrets first).
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
