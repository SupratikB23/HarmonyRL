import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonyrl.inference import generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--no_audio", action="store_true")
    ap.add_argument("--use_diffusers", action="store_true")
    a = ap.parse_args()
    print("Saved:", generate(ckpt=a.ckpt, ckpt_dir=a.ckpt_dir, out_dir=a.output_dir,
                             n_samples=a.n_samples, max_new_tokens=a.max_new_tokens,
                             temperature=a.temperature, top_p=a.top_p,
                             render_audio=not a.no_audio, use_diffusers=a.use_diffusers))
