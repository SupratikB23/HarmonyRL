import os, glob, random
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import pretty_midi

from harmonyrl.midi_utils import midi_to_tokens

class MIDITokenDataset(Dataset):
    def __init__(self, root: str, max_seq_len: int = 2048, split: str = "train", train_ratio: float = 0.95):
        midi_paths: List[str] = sorted(glob.glob(os.path.join(root, "**/*.mid*"), recursive=True))
        if not midi_paths:
            raise FileNotFoundError(f"No MIDI files under: {root}")
        random.seed(42)
        random.shuffle(midi_paths)
        n_train = int(len(midi_paths) * train_ratio)
        self.paths = midi_paths[:n_train] if split == "train" else midi_paths[n_train:]
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pm = pretty_midi.PrettyMIDI(self.paths[idx])
        tokens = midi_to_tokens(pm, self.max_seq_len)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:],  dtype=torch.long)
        return x, y

def make_loaders(root: str, max_seq_len: int, batch_size: int = 8, train_ratio: float = 0.95):
    train_ds = MIDITokenDataset(root, max_seq_len, "train", train_ratio)
    val_ds   = MIDITokenDataset(root, max_seq_len, "val",   train_ratio)

    def collate(batch):
        xs, ys = zip(*batch)
        lens = [len(x) for x in xs]
        maxlen = max(lens)
        pad = 0
        X = torch.full((len(xs), maxlen), pad, dtype=torch.long)
        Y = torch.full((len(xs), maxlen), pad, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            L = len(x)
            X[i, :L] = x
            Y[i, :L] = y
        return X, Y, torch.tensor(lens)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader
