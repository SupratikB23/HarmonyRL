import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonyrl.training.rl import train_rl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rl_config.yaml")
    print("Saved:", train_rl(ap.parse_args().config))
