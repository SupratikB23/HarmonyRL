import os, yaml, tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from harmonyrl.datasets import make_loaders
from harmonyrl.midi_utils import vocab_size
from harmonyrl.models.lstm import LSTMModel
from harmonyrl.utils.logging import get_logger

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        n_class = pred.size(-1)

        mask = target != self.ignore_index
        target = target[mask]
        pred = pred[mask]

        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (n_class - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

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

    opt = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, verbose=True
    )

    if cfg["train"].get("label_smoothing", 0.0) > 0:
        loss_fn = LabelSmoothingLoss(
            smoothing=cfg["train"]["label_smoothing"], ignore_index=0
        )
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler(enabled=(device == "cuda"))

    def run_epoch(loader, is_train=True):
        model.train(is_train)
        total, count = 0.0, 0
        with torch.set_grad_enabled(is_train):
            for X, Y, _ in tqdm.tqdm(loader):
                X, Y = X.to(device), Y.to(device)
                with autocast(enabled=(device == "cuda")):
                    logits, _ = model(X)
                    loss = loss_fn(
                        logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
                if is_train:
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["train"]["clip_grad_norm"])
                    scaler.step(opt)
                    scaler.update()
                total += loss.item() * X.size(0)
                count += X.size(0)
        return total / max(1, count)

    best = float("inf")
    patience, bad_epochs = cfg["train"].get("patience", 5), 0

    for epoch in range(cfg["train"]["epochs"]):
        tr_loss = run_epoch(train_loader, True)
        va_loss = run_epoch(val_loader, False)

        scheduler.step(va_loss)

        log.info(f"epoch {epoch+1}/{cfg['train']['epochs']} "
                 f"| train={tr_loss:.4f} | val={va_loss:.4f}")

        if va_loss < best:
            os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
            ckpt_path = os.path.join(cfg["train"]["ckpt_dir"], "lstm_supervised.pt")
            torch.save(model.state_dict(), ckpt_path)
            best = va_loss
            bad_epochs = 0
            log.info(f"✅ checkpoint saved: {ckpt_path}")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            log.info("⏹ Early stopping triggered.")
            break