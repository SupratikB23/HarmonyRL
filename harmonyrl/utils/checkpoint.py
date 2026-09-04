from typing import Dict, Optional

import torch

from harmonyrl.midi_utils import vocab_size
from harmonyrl.models import LSTMModel, MusicTransformer

ARCHS = {"lstm": LSTMModel, "transformer": MusicTransformer}


def build_model(model_cfg: Dict):
    cfg = dict(model_cfg)
    arch = cfg.pop("arch", "lstm")
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch '{arch}', expected one of {sorted(ARCHS)}")
    return ARCHS[arch](vocab_size(), **cfg), arch, cfg


def save_checkpoint(path: str, model, model_cfg: Dict, extra: Optional[Dict] = None):
    torch.save({"model_config": dict(model_cfg), "vocab_size": vocab_size(),
                "state_dict": model.state_dict(), **(extra or {})}, path)


def load_model(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model_config" not in ck:
        raise ValueError(f"{path} predates the current checkpoint format; retrain.")
    if ck.get("vocab_size") != vocab_size():
        raise ValueError(f"{path} was trained on a vocab of {ck.get('vocab_size')}, "
                         f"current tokenizer has {vocab_size()}.")
    model, _, _ = build_model(ck["model_config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck


def device_of(prefer: Optional[str] = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"
