import torch
import torch.nn as nn
import torch.nn.functional as F

from harmonyrl.midi_utils import BOS, EOS, PAD
from harmonyrl.models.sampling import sample_step


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_len: int):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_len).float(), inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, offset: int):
        T = x.size(-2)
        cos = self.cos[offset:offset + T].to(x.dtype)
        sin = self.sin[offset:offset + T].to(x.dtype)
        return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, d_model, nhead, dropout, rotary):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rotary = rotary

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.nhead, self.head_dim).transpose(1, 2) for t in (q, k, v))

        offset = cache[0].size(2) if cache is not None else 0
        q, k = self.rotary(q, offset), self.rotary(k, offset)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        # is_causal only expresses the no-cache case; with a cache the query at
        # position offset+i may attend to keys 0..offset+i, so build that mask.
        attn_mask, is_causal = None, False
        if T > 1:
            if offset == 0:
                is_causal = True
            else:
                rows = torch.arange(T, device=x.device) + offset
                cols = torch.arange(k.size(2), device=x.device)
                attn_mask = cols[None, :] <= rows[:, None]
        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0)
        h = h.transpose(1, 2).reshape(B, T, C)
        return self.out(h), (k, v)


class Block(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout, rotary):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, nhead, dropout, rotary)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, cache=None):
        h, cache = self.attn(self.ln1(x), cache)
        x = x + h
        return x + self.ff(self.ln2(x)), cache


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.2, max_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        rotary = Rotary(d_model // nhead, max_len)
        self.layers = nn.ModuleList(
            [Block(d_model, nhead, dim_feedforward, dropout, rotary) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init)
        # _init re-randomises the tied embed/head matrix (it is visited twice, once
        # as an Embedding and once as a Linear), wiping the padding_idx zero row.
        with torch.no_grad():
            self.embed.weight[PAD].zero_()

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def features(self, x, caches=None):
        h = self.drop(self.embed(x))
        new_caches = []
        for i, layer in enumerate(self.layers):
            h, c = layer(h, None if caches is None else caches[i])
            new_caches.append(c)
        return self.norm(h), new_caches

    def forward(self, x, caches=None):
        h, caches = self.features(x, caches)
        return self.head(h), caches

    @torch.no_grad()
    def sample(self, batch_size=1, max_new_tokens=512, temperature=1.0, top_p=0.95, device="cpu"):
        was_training = self.training
        self.eval()
        tok = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
        out, caches = [tok], None
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(min(max_new_tokens, self.max_len - 1)):
            logits, caches = self.forward(tok, caches)
            tok = sample_step(logits[:, -1, :], temperature, top_p)
            tok = tok.masked_fill(done.unsqueeze(1), PAD)
            done |= tok.squeeze(1) == EOS
            out.append(tok)
            if bool(done.all()):
                break
        self.train(was_training)
        return torch.cat(out, dim=1)
