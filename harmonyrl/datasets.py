import glob
import hashlib
import os
import re
from typing import List, Optional

import numpy as np
import pretty_midi
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from harmonyrl.midi_utils import (PAD, STEPS_PER_BAR, VOCAB_SIZE, midi_to_tokens)

MAX_FILE_TOKENS = 200_000
MIN_CHUNK_TOKENS = 128


def _group_of(path: str) -> str:
    """Movements of one recording share a group so they can't straddle the split."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.split(r"--AUDIO|_wav|--\d+$", stem)[0]


def _build_cache(paths: List[str], cache_path: str):
    tokens, offsets, groups = [], [0], []
    for p in tqdm.tqdm(paths, desc="tokenizing"):
        try:
            seq = midi_to_tokens(pretty_midi.PrettyMIDI(p), MAX_FILE_TOKENS)
        except Exception:
            continue
        if len(seq) < MIN_CHUNK_TOKENS:
            continue
        tokens.append(np.asarray(seq, dtype=np.int16))
        offsets.append(offsets[-1] + len(seq))
        groups.append(_group_of(p))
    if not groups:
        raise RuntimeError("No MIDI file could be tokenized.")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, tokens=np.concatenate(tokens),
             offsets=np.asarray(offsets, dtype=np.int64), groups=np.asarray(groups))


def _load_cache(root: str, cache_dir: str, rebuild: bool):
    paths = sorted(glob.glob(os.path.join(root, "**", "*.mid*"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No MIDI files under: {root}")
    # The tokenizer layout is part of the key: otherwise changing the vocabulary
    # silently reuses a cache whose ids mean something else.
    sig = f"{VOCAB_SIZE}:{STEPS_PER_BAR}:{MAX_FILE_TOKENS}|" + "|".join(paths)
    key = hashlib.md5(sig.encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"tokens_{key}.npz")
    if rebuild or not os.path.exists(cache_path):
        _build_cache(paths, cache_path)
    return np.load(cache_path, allow_pickle=False)


class MIDITokenDataset(Dataset):
    """Fixed-length chunks over the whole corpus, not one truncated sample per file."""

    def __init__(self, root: str, max_seq_len: int = 1024, split: str = "train",
                 train_ratio: float = 0.95, stride: Optional[int] = None,
                 cache_dir: str = ".cache", rebuild_cache: bool = False):
        cache = _load_cache(root, cache_dir, rebuild_cache)
        tokens, offsets, groups = cache["tokens"], cache["offsets"], cache["groups"]

        uniq = sorted(set(groups.tolist()))
        rng = np.random.RandomState(42)
        rng.shuffle(uniq)
        n_train = max(1, int(len(uniq) * train_ratio))
        keep = set(uniq[:n_train] if split == "train" else uniq[n_train:])

        self.tokens = tokens
        self.window = max_seq_len + 1
        stride = stride or max_seq_len // 2
        self.index = []
        for i, g in enumerate(groups):
            if g not in keep:
                continue
            start, end = int(offsets[i]), int(offsets[i + 1])
            for s in range(start, max(start + 1, end - MIN_CHUNK_TOKENS + 1), stride):
                self.index.append((s, min(s + self.window, end)))
        if not self.index:
            raise RuntimeError(f"Split '{split}' is empty; lower train_ratio.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        s, e = self.index[idx]
        chunk = torch.from_numpy(self.tokens[s:e].astype(np.int64))
        out = torch.full((self.window,), PAD, dtype=torch.long)
        out[: len(chunk)] = chunk
        return out[:-1], out[1:]


def make_loaders(root: str, max_seq_len: int, batch_size: int = 8, train_ratio: float = 0.95,
                 num_workers: int = 0, **kwargs):
    train_ds = MIDITokenDataset(root, max_seq_len, "train", train_ratio, **kwargs)
    val_ds = MIDITokenDataset(root, max_seq_len, "val", train_ratio, **kwargs)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=False)
    return (DataLoader(train_ds, shuffle=True, **common),
            DataLoader(val_ds, shuffle=False, **common))
