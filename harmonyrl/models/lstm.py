import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=512, hidden=768, layers=3, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.drop(out)
        return self.head(out), hidden

    @torch.no_grad()
    def sample(self, bos=1, max_new_tokens=512, temperature=1.0, top_p=0.95, device="cpu"):
        self.eval()
        x = torch.tensor([[bos]], dtype=torch.long, device=device)
        hidden = None
        out_tokens = [bos]
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(x, hidden)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)
            
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cum > top_p).float().argmax(dim=-1)
            mask = torch.arange(probs.size(-1), device=device)[None, :] > cutoff[:, None]
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(sorted_probs, 1)
            tok = sorted_idx.gather(-1, idx).item()

            out_tokens.append(tok)
            if tok == 2: break
            x = torch.tensor([[tok]], dtype=torch.long, device=device)
        return out_tokens