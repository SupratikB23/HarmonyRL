import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        self.register_buffer("pos_emb", emb, persistent=False)
    def forward(self, x):
        T = x.size(1)
        return self.pos_emb[:T, :]

def apply_rotary(x, rope):
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    sin, cos = rope[..., : d // 2], rope[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)[0]
        x = x + self.ff(self.ln2(x))
        return x

class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.2,
        max_len=4096,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.rope = RotaryPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) * math.sqrt(self.d_model)
        rope = self.rope(h).to(h.device) 
        h = apply_rotary(h, rope)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        return self.head(h)

    @torch.no_grad()
    def sample(
        self,
        bos=1,
        max_new_tokens=512,
        temperature=1.0,
        top_p=0.9,
        device="cpu",
    ):
        self.eval()
        x = torch.tensor([[bos]], dtype=torch.long, device=device)
        out = [bos]

        for _ in range(max_new_tokens):
            logits = self.forward(x)[:, -1, :] / max(1e-6, temperature)
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            idx = torch.multinomial(sorted_probs, 1)
            tok = sorted_idx.gather(-1, idx).item()

            out.append(tok)
            if tok == 2: 
                break
            x = torch.cat([x, torch.tensor([[tok]], device=device)], dim=1)
        return out