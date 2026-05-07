# Contributing to HarmonyRL

Thanks for your interest in improving HarmonyRL! This project is an active research experiment, so small, focused contributions are preferred.

## Getting Started

1. **Fork & clone**
   ```bash
   git clone https://github.com/<your-username>/HarmonyRL.git
   cd HarmonyRL
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   # venv\Scripts\activate       # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) Install in editable mode**
   ```bash
   pip install -e .
   ```
5. **Dataset**
   Download the MAESTRO dataset and place it under `data/maestro/` as described in the README.

## Running the Project

- **Supervised pretraining**
  ```bash
  python scripts/train_supervised.py --config configs/supervised_config.yaml
  ```
- **RL fine-tuning**
  ```bash
  python scripts/train_rl.py --config configs/rl_config.yaml
  ```
- **Inference**
  ```bash
  python scripts/infer.py --ckpt checkpoints/best_model.pt --output_dir outputs/
  ```

## Development Guidelines

- Keep changes focused and aligned with the experimental goals described in the README.
- Follow existing code style and naming conventions in the touched files.
- Update documentation and config examples when behavior or defaults change.
- If you add new dependencies, explain why they are needed and keep them minimal.

## Tests & Validation

There is currently no automated test suite. If you introduce tests, document how to run them in your PR description and ensure they pass.

## Submitting Changes

1. Create a feature branch from `main`.
2. Make your changes with clear, descriptive commit messages.
3. Open a pull request explaining **what** changed and **why**, and include:
   - Relevant training/inference commands you ran
   - Config changes and expected behavior updates

## Reporting Issues

When filing a bug report, please include:
- Your OS and Python version
- The command you ran and its configuration
- Minimal reproduction steps
- Logs or error tracebacks

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
