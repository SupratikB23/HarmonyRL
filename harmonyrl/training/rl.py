import copy
import os

import numpy as np
import torch
import torch.nn as nn
import tqdm
import yaml
from torch.optim import AdamW

from harmonyrl.midi_utils import EOS
from harmonyrl.rewards import reward_parts, combine_rewards
from harmonyrl.utils import device_of, get_logger, load_model, save_checkpoint


class ActorCritic(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.value = nn.Linear(backbone.head.in_features, 1)
        nn.init.zeros_(self.value.bias)

    def forward(self, x):
        h = self.backbone.features(x)[0]
        return self.backbone.head(h), self.value(h).squeeze(-1)


def gather_logprobs(logits, targets):
    return torch.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def masked_entropy(logits, mask):
    logp = torch.log_softmax(logits, -1)
    ent = -(logp.exp() * logp).sum(-1)
    return (ent * mask).sum() / mask.sum().clamp_min(1)


def sequence_rewards(seq, weights):
    rewards, parts = [], []
    for row in seq.tolist():
        tokens = row[: row.index(EOS) + 1] if EOS in row else row
        p = reward_parts(tokens)
        parts.append(p)
        rewards.append(combine_rewards(p, weights))
    return torch.tensor(rewards, dtype=torch.float32), parts


def live_mask(targets):
    """1 up to and including the first EOS, 0 after it.

    `targets != PAD` is NOT equivalent: PAD is a sampleable id, so a stray PAD
    would punch a hole in the middle of an episode and make `mask.sum(1)` a
    wrong episode length.
    """
    ended = (targets == EOS).cumsum(1) - (targets == EOS).long()
    return (ended == 0).float()


def compute_gae(rewards, values, mask, gamma, lam):
    T = rewards.size(1)
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else torch.zeros_like(values[:, 0])
        next_nonterm = mask[:, t + 1] if t + 1 < T else torch.zeros_like(mask[:, 0])
        delta = rewards[:, t] + gamma * next_v * next_nonterm - values[:, t]
        last = delta + gamma * lam * next_nonterm * last
        adv[:, t] = last
    return adv * mask


def train_rl(config_path: str = "configs/rl_config.yaml"):
    log = get_logger("rl")
    cfg = yaml.safe_load(open(config_path, "r"))
    rc = cfg["rl"]
    torch.manual_seed(cfg["seed"])
    device = device_of(rc.get("device"))

    ckpt_dir = cfg["train"]["ckpt_dir"]
    sup_ckpt = os.path.join(ckpt_dir, cfg["train"]["init_from"])
    if not os.path.exists(sup_ckpt):
        raise FileNotFoundError(f"Need a supervised checkpoint at {sup_ckpt}; run train_supervised first.")
    backbone, meta = load_model(sup_ckpt, device)
    log.info(f"loaded {sup_ckpt} (val_loss={meta.get('val_loss')})")

    model = ActorCritic(backbone).to(device)
    model.eval()  # dropout off, so the PPO ratio starts at exactly 1
    ref = copy.deepcopy(backbone).to(device).eval()
    ref.requires_grad_(False)

    opt = AdamW(model.parameters(), lr=float(rc["lr"]), weight_decay=0.0)
    weights = cfg["reward"]["weights"]
    gamma, lam = rc.get("gamma", 1.0), rc.get("gae_lambda", 0.95)
    clip, kl_coef = rc.get("clip_range", 0.2), rc.get("kl_coef", 0.05)
    reward_log = []

    for it in tqdm.tqdm(range(rc["iterations"])):
        seq = backbone.sample(batch_size=rc["batch_size"], max_new_tokens=rc["rollout_len"],
                              temperature=rc.get("temperature", 1.0),
                              top_p=rc.get("top_p", 0.95), device=device)
        inputs, targets = seq[:, :-1], seq[:, 1:]
        mask = live_mask(targets)
        if mask.sum() == 0:
            continue

        with torch.no_grad():
            logits, values = model(inputs)
            logp_old = gather_logprobs(logits, targets)
            logp_ref = gather_logprobs(ref(inputs)[0], targets)
            values = values * mask

        R, parts = sequence_rewards(seq, weights)
        R = R.to(device)
        reward_log.append(R.mean().item())

        # per-token KL shaping, sequence reward paid at the final real token
        token_rewards = -kl_coef * (logp_old - logp_ref) * mask
        last_idx = mask.sum(1).long() - 1
        token_rewards[torch.arange(seq.size(0), device=device), last_idx] += R

        adv = compute_gae(token_rewards, values, mask, gamma, lam)
        returns = adv + values
        flat = adv[mask.bool()]
        # unbiased std is NaN for a single live token; fall back to no scaling
        std = flat.std() if flat.numel() > 1 else torch.zeros((), device=device)
        adv = (adv - flat.mean()) / (std + 1e-8) * mask

        for _ in range(rc.get("ppo_epochs", 4)):
            logits, values_new = model(inputs)
            logp = gather_logprobs(logits, targets)
            ratio = (logp - logp_old).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            pg_loss = (pg * mask).sum() / mask.sum()
            v_loss = (((values_new - returns) ** 2) * mask).sum() / mask.sum()
            entropy = masked_entropy(logits, mask)
            loss = pg_loss + rc.get("value_coef", 0.5) * v_loss - rc["entropy_coef"] * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rc.get("clip_grad_norm", 1.0))
            opt.step()

        if (it + 1) % rc.get("log_interval", 10) == 0:
            kl = ((logp_old - logp_ref) * mask).sum() / mask.sum()
            avg = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
            log.info(f"it {it + 1} | R={R.mean():.3f} (avg100={np.mean(reward_log[-100:]):.3f}) "
                     f"| kl={kl:.4f} | ent={entropy:.3f} | v={v_loss:.3f} | "
                     + " ".join(f"{k}={v:.2f}" for k, v in avg.items()))

        if (it + 1) % rc.get("save_interval", 100) == 0:
            path = os.path.join(ckpt_dir, f"{meta['model_config'].get('arch', 'lstm')}_rl.pt")
            save_checkpoint(path, backbone, meta["model_config"], {"iteration": it + 1, "value_head": model.value.state_dict()})
            log.info(f"saved {path}")

    path = os.path.join(ckpt_dir, f"{meta['model_config'].get('arch', 'lstm')}_rl.pt")
    save_checkpoint(path, backbone, meta["model_config"], {"iteration": rc["iterations"], "value_head": model.value.state_dict()})
    return path
