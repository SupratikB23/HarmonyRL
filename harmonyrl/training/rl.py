import os, yaml, tqdm, torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from harmonyrl.midi_utils import vocab_size
from harmonyrl.models.lstm import LSTMModel
from harmonyrl.utils.logging import get_logger
from harmonyrl.utils.evaluation import simple_harmony_reward

class EMA:
    def __init__(self, beta=0.95):
        self.beta, self.v = beta, None
    def update(self, x):
        self.v = x if self.v is None else self.beta * self.v + (1 - self.beta) * x
        return self.v

def compute_logprobs(model, tokens, device):
    x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x)
    logp = F.log_softmax(logits, dim=-1)
    chosen = logp.gather(-1, y.unsqueeze(-1)).squeeze(0) 
    return chosen

def train_rl(config_path: str = "configs/rl_config.yaml"):
    log = get_logger("rl")
    cfg = yaml.safe_load(open(config_path, "r"))
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(
        vocab_size(),
        cfg["model"]["embed_dim"],
        cfg["model"]["hidden"],
        cfg["model"]["layers"],
        cfg["model"]["dropout"],
    ).to(device)

    sup_ck = os.path.join(cfg["train"]["ckpt_dir"], "lstm_supervised.pt")
    if os.path.exists(sup_ck):
        model.load_state_dict(torch.load(sup_ck, map_location=device))
        log.info("Loaded supervised checkpoint.")

    opt = AdamW(model.parameters(), lr=float(cfg["rl"]["lr"]), weight_decay=1e-4)
    baseline = EMA(cfg["rl"]["baseline_beta"])
    entropy_coef = cfg["rl"]["entropy_coef"]

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    reward_history = []

    for ep in tqdm.tqdm(range(cfg["rl"]["episodes"])):
        tokens = model.sample(
            max_new_tokens=cfg["rl"]["rollout_len"],
            device=device,
            temperature=1.0,  
            top_p=0.9,
        )

        # === Compute reward ===
        R = simple_harmony_reward(tokens)
        reward_history.append(R)

        # === Log-probs ===
        logps = compute_logprobs(model, tokens, device)  # [T]
        logp_traj = logps.mean()  # average trajectory log-prob

        # === Entropy bonus ===
        x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()

        baseline_val = baseline.update(R)
        adv = R - baseline_val
        adv /= (torch.tensor(reward_history[-100:]).std() + 1e-6)  # normalize

        loss = -(adv * logp_traj) - entropy_coef * entropy
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if (ep + 1) % cfg["train"]["log_interval"] == 0:
            log.info(
                f"ep {ep+1} | R={R:.3f} | baseline={baseline_val:.3f} "
                f"| adv={adv:.3f} | ent={entropy.item():.3f} | loss={loss.item():.3f}"
            )

        if (ep + 1) % cfg["train"]["save_interval"] == 0:
            os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
            ckpt_path = os.path.join(cfg["train"]["ckpt_dir"], f"lstm_rl_ep{ep+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            log.info(f"Saved checkpoint: {ckpt_path}")