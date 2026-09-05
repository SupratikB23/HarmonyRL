import torch
import torch.nn as nn

from harmonyrl.midi_utils import BOS, EOS, PAD
from harmonyrl.models.sampling import sample_step


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=512, hidden=512, layers=3, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        # project back to embed_dim so tying stays valid when hidden != embed_dim
        self.proj = nn.Linear(hidden, embed_dim, bias=False) if hidden != embed_dim else nn.Identity()
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()

    def features(self, x, state=None):
        out, state = self.lstm(self.embed(x), state)
        return self.proj(self.drop(self.norm(out))), state

    def forward(self, x, state=None):
        h, state = self.features(x, state)
        return self.head(h), state

    @torch.no_grad()
    def sample(self, batch_size=1, max_new_tokens=512, temperature=1.0, top_p=0.95, device="cpu"):
        was_training = self.training
        self.eval()
        tok = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
        out, state = [tok], None
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            logits, state = self.forward(tok, state)
            tok = sample_step(logits[:, -1, :], temperature, top_p)
            tok = tok.masked_fill(done.unsqueeze(1), PAD)
            done |= tok.squeeze(1) == EOS
            out.append(tok)
            if bool(done.all()):
                break
        self.train(was_training)
        return torch.cat(out, dim=1)
