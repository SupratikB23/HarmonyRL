import math
import os

import torch
import torch.nn as nn
import tqdm
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from harmonyrl.datasets import make_loaders
from harmonyrl.midi_utils import PAD
from harmonyrl.utils import build_model, device_of, get_logger, save_checkpoint


def _lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train_supervised(config_path: str = "configs/supervised_config.yaml"):
    log = get_logger("supervised")
    cfg = yaml.safe_load(open(config_path, "r"))
    tcfg = cfg["train"]
    torch.manual_seed(cfg["seed"])

    train_loader, val_loader = make_loaders(
        root=cfg["data"]["root"],
        max_seq_len=cfg["data"]["max_seq_len"],
        batch_size=tcfg["batch_size"],
        train_ratio=cfg["data"]["train_ratio"],
        num_workers=cfg["data"].get("num_workers", 0),
    )
    log.info(f"train chunks={len(train_loader.dataset)} val chunks={len(val_loader.dataset)}")

    device = device_of(tcfg.get("device"))
    model, arch, _ = build_model(cfg["model"])
    model = model.to(device)
    log.info(f"{arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params on {device}")

    opt = AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.01)))
    total_steps = len(train_loader) * tcfg["epochs"]
    warmup = min(tcfg.get("warmup_steps", 500), max(1, total_steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_lambda(s, warmup, total_steps))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=tcfg.get("label_smoothing", 0.0))
    # model selection / early stopping must compare true NLL, not the smoothed
    # training objective (smoothing adds a constant floor to the reported loss).
    eval_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    scaler = GradScaler(device, enabled=(device == "cuda"))

    def run_epoch(loader, train: bool):
        model.train(train)
        total, count = 0.0, 0
        for X, Y in tqdm.tqdm(loader, leave=False):
            X, Y = X.to(device), Y.to(device)
            n = int((Y != PAD).sum())
            if n == 0:  # an all-padding batch makes cross_entropy return NaN
                continue
            with torch.set_grad_enabled(train), autocast(device, enabled=(device == "cuda")):
                logits = model(X)[0]
                fn = loss_fn if train else eval_loss_fn
                loss = fn(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip_grad_norm"])
                scaler.step(opt)
                scaler.update()
                sched.step()
            total += loss.item() * n
            count += n
        return total / max(1, count)

    os.makedirs(tcfg["ckpt_dir"], exist_ok=True)
    ckpt_path = os.path.join(tcfg["ckpt_dir"], f"{arch}_supervised.pt")
    best, bad = float("inf"), 0

    for epoch in range(tcfg["epochs"]):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        log.info(f"epoch {epoch + 1}/{tcfg['epochs']} | train {tr:.4f} "
                 f"| val {va:.4f} | val ppl {math.exp(min(20, va)):.2f}")

        if va < best:
            best, bad = va, 0
            save_checkpoint(ckpt_path, model, cfg["model"], {"val_loss": va, "epoch": epoch})
            log.info(f"saved {ckpt_path}")
        else:
            bad += 1
            if bad >= tcfg.get("patience", 5):
                log.info("early stopping")
                break
    return ckpt_path
