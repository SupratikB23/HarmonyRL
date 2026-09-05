# Contributing to HarmonyRL

HarmonyRL is an active research experiment, so small, focused contributions are preferred.

## Setup

```bash
git clone https://github.com/<your-username>/HarmonyRL.git
cd HarmonyRL
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                  # optional; scripts run without it
```

Download [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) into `data/maestro/`.
Training needs a GPU; see [`notebooks/`](notebooks/) for molab and Kaggle notebooks.

## Running things

```bash
python scripts/train_supervised.py --config configs/supervised_config.yaml
python scripts/train_rl.py         --config configs/rl_config.yaml
python scripts/infer.py --n_samples 4 --output_dir outputs/
python scripts/to_mp3.py                     # needs: pip install lameenc
```

## Tests

```bash
pytest
```

46 tests must pass before a pull request. They exist because most of them were written in
response to a real bug, so treat a failure as a genuine regression rather than a flaky
check. If you change behaviour, add a test that would have caught the old behaviour.

Two areas deserve extra care:

- **Tokenizer.** `midi_to_tokens` and `tokens_to_midi` must round-trip note count,
  polyphony, timing and velocity. A silent asymmetry here corrupts every downstream stage.
- **Reward functions.** Any new term must not be maximizable by degenerate output.
  `tests/test_rewards.py` asserts that a single repeated note scores worse than varied,
  in-key material.

## Guidelines

- Keep changes focused, and match the style of the file you are touching.
- Update configs, docs and the README when defaults or behaviour change.
- Justify new dependencies. The runtime set is deliberately small, and packages that can
  drag in a CPU build of torch on a GPU host are a particular problem.
- Report measured numbers, not estimates. If you claim something is faster or better, say
  what you ran.

## Pull requests

Branch from `main`, then open a PR describing what changed and why. Include the commands
you ran, any config changes, and the test result. For anything affecting training quality,
include validation perplexity or the relevant evaluation metrics from
`harmonyrl/utils/evaluation.py`.

## Bug reports

Include your OS and Python version, the exact command and config, minimal reproduction
steps, and the full traceback.

## License

Contributions are licensed under Apache 2.0.
