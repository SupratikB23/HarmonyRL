import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonyrl.training.supervised import train_supervised

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/supervised_config.yaml")
    print("Saved:", train_supervised(ap.parse_args().config))
