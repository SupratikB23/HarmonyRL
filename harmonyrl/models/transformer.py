import math, torch
import torch.nn as nn

class SmallTransformer(nn.Module):
    """
    Lean Transformer for token generation (optional alternative to LSTM).
    """
    def __init__(self, vocab_size: int, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_len=4096):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        x = self.tr(x)
        return self.head(x)

    @torch.no_grad()
    def sample(self, bos=1, max_new_tokens=512, temperature=1.0, top_p=0.95, device="cpu"):
        self.eval()
        x = torch.tensor([[bos]], dtype=torch.long, device=device)
        out = [bos]
        for _ in range(max_new_tokens):
            logits = self.forward(x)[:, -1, :] / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cum > top_p).float().argmax(dim=-1)
            mask = torch.arange(probs.size(-1), device=device)[None, :] > cutoff[:, None]
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(sorted_probs, 1)
            tok = sorted_idx.gather(-1, idx).item()
            out.append(tok)
            if tok == 2: break
            x = torch.cat([x, torch.tensor([[tok]], device=device)], dim=1)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)
