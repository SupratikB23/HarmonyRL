import os, yaml, tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW

from harmonyrl.datasets import make_loaders
from harmonyrl.midi_utils import vocab_size
from harmonyrl.models.lstm import LSTMModel
from harmonyrl.utils.logging import get_logger

def train_supervised(config_path: str = "configs/supervised_config.yaml"):
    log = get_logger("supervised")
    cfg = yaml.safe_load(open(config_path, "r"))
    torch.manual_seed(cfg["seed"])

    train_loader, val_loader = make_loaders(
        root=cfg["data"]["root"],
        max_seq_len=cfg["data"]["max_seq_len"],
        batch_size=cfg["train"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(
        vocab_size=vocab_size(),
        embed_dim=cfg["model"]["embed_dim"],
        hidden=cfg["model"]["hidden"],
        layers=cfg["model"]["layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    opt = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def run_epoch(loader, is_train=True):
        model.train(is_train)
        total = 0.0
        count = 0
        for X, Y, _ in tqdm.tqdm(loader):
            X, Y = X.to(device), Y.to(device)
            logits, _ = model(X)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            if is_train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_grad_norm"])
                opt.step()
            total += loss.item() * X.size(0)
            count += X.size(0)
        return total / max(1, count)

    best = 1e9
    for epoch in range(cfg["train"]["epochs"]):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        log.info(f"epoch {epoch}: train={tr:.4f} | val={va:.4f}")
        if va < best:
            os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg["train"]["ckpt_dir"], "lstm_supervised.pt"))
            best = va
            log.info("checkpoint saved.")
