import os, yaml, tqdm, torch
from torch.optim import Adam

from harmonyrl.midi_utils import vocab_size, tokens_to_midi, synth_audio
from harmonyrl.models.lstm import LSTMModel
from harmonyrl.utils.logging import get_logger
from harmonyrl.utils.evaluation import simple_harmony_reward

class EMA:
    def __init__(self, beta=0.95): self.beta, self.v = beta, None
    def update(self, x):
        self.v = x if self.v is None else self.beta*self.v + (1-self.beta)*x
        return self.v

def traj_logprob(model, tokens, device):
    x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(tokens[1:],  dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x)
    logp = torch.log_softmax(logits, dim=-1)
    chosen = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [1, T]
    return chosen.mean()

def train_rl(config_path: str = "configs/rl_config.yaml"):
    log = get_logger("rl")
    cfg = yaml.safe_load(open(config_path, "r"))
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LSTMModel(vocab_size(), cfg["model"]["embed_dim"], cfg["model"]["hidden"],
                      cfg["model"]["layers"], cfg["model"]["dropout"]).to(device)

    sup_ck = os.path.join(cfg["train"]["ckpt_dir"], "lstm_supervised.pt")
    if os.path.exists(sup_ck):
        model.load_state_dict(torch.load(sup_ck, map_location=device))
        log.info("Loaded supervised checkpoint.")

    opt = Adam(model.parameters(), lr=float(cfg["rl"]["lr"]))
    baseline = EMA(cfg["rl"]["baseline_beta"])
    entropy_coef = cfg["rl"]["entropy_coef"]

    model.train()
    for ep in tqdm.tqdm(range(cfg["rl"]["episodes"])):
        # sample sequence
        tokens = model.sample(max_new_tokens=cfg["rl"]["rollout_len"], device=device, temperature=0.9, top_p=0.95)

        # compute reward (symbolic simple harmony; can extend to audio-based)
        R = simple_harmony_reward(tokens)

        # policy gradient
        lp = traj_logprob(model, tokens, device)
        # quick entropy term from last step
        x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        logp = torch.log_softmax(logits, dim=-1)
        entropy = -(logp * torch.exp(logp)).sum(dim=-1).mean()

        adv = R - baseline.update(R)
        loss = -(adv * lp) - entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (ep + 1) % 10 == 0:
            log.info(f"ep {ep+1} | R={R:.3f} | lp={lp.item():.3f} | ent={entropy.item():.3f} | loss={loss.item():.3f}")
            os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg["train"]["ckpt_dir"], "lstm_rl.pt"))
