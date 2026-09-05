import pytest
import torch

from harmonyrl.midi_utils import BOS, EOS, PAD, vocab_size
from harmonyrl.models import LSTMModel, MusicTransformer
from harmonyrl.utils import build_model, load_model, save_checkpoint

CONFIGS = [
    {"arch": "lstm", "embed_dim": 32, "hidden": 48, "layers": 2, "dropout": 0.0},
    {"arch": "transformer", "d_model": 32, "nhead": 4, "num_layers": 2,
     "dim_feedforward": 64, "dropout": 0.0, "max_len": 128},
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_forward_and_sample_shapes(cfg):
    model, _, _ = build_model(cfg)
    logits = model(torch.randint(4, 100, (2, 16)))[0]
    assert logits.shape == (2, 16, vocab_size())
    seq = model.sample(batch_size=3, max_new_tokens=20)
    assert seq.shape[0] == 3 and seq.shape[1] <= 21


def test_transformer_is_causal():
    model = MusicTransformer(vocab_size(), d_model=32, nhead=4, num_layers=2,
                             dim_feedforward=64, dropout=0.0).eval()
    x = torch.randint(4, 100, (2, 12))
    a = model(x)[0]
    x2 = x.clone()
    x2[:, 6:] = torch.randint(4, 100, (2, 6))
    assert torch.allclose(a[:, :6], model(x2)[0][:, :6], atol=1e-5)


def test_kv_cache_matches_full_forward():
    model = MusicTransformer(vocab_size(), d_model=32, nhead=4, num_layers=2,
                             dim_feedforward=64, dropout=0.0).eval()
    x = torch.randint(4, 100, (2, 10))
    full = model(x)[0]
    cache, steps = None, []
    for t in range(x.size(1)):
        logits, cache = model(x[:, t:t + 1], cache)
        steps.append(logits)
    assert torch.allclose(full, torch.cat(steps, 1), atol=1e-4)


def test_lstm_tying_survives_hidden_mismatch():
    model = LSTMModel(vocab_size(), embed_dim=32, hidden=48, layers=1)
    assert model.head.weight is model.embed.weight


@pytest.mark.parametrize("cfg", CONFIGS)
def test_checkpoint_roundtrip(cfg, tmp_path):
    model, _, _ = build_model(cfg)
    path = tmp_path / "m.pt"
    save_checkpoint(str(path), model, cfg)
    loaded, meta = load_model(str(path))
    assert meta["model_config"] == cfg
    x = torch.randint(4, 100, (1, 8))
    model.eval(), loaded.eval()
    assert torch.allclose(model(x)[0], loaded(x)[0], atol=1e-6)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_sample_pads_after_eos(cfg):
    model, _, _ = build_model(cfg)
    seq = model.sample(batch_size=4, max_new_tokens=30)
    for row in seq.tolist():
        if EOS in row:
            assert set(row[row.index(EOS) + 1:]) <= {PAD}


@pytest.mark.parametrize("cfg", CONFIGS)
def test_sampling_never_emits_pad_or_bos(cfg):
    """PAD/BOS logits are unconstrained by the loss, so they must be masked at
    sampling time -- a stray PAD mid-episode corrupts the RL episode mask."""
    torch.manual_seed(0)
    model, _, _ = build_model(cfg)
    seq = model.sample(batch_size=6, max_new_tokens=120)
    for row in seq.tolist():
        body = row[: row.index(EOS) + 1] if EOS in row else row
        assert PAD not in body[1:]
        assert BOS not in body[1:]


def test_cached_multi_token_prefill_matches_full_forward():
    """The cached path must stay causal for a prefill longer than one token."""
    model = MusicTransformer(vocab_size(), d_model=32, nhead=4, num_layers=2,
                             dim_feedforward=64, dropout=0.0).eval()
    x = torch.randint(4, 100, (2, 12))
    full = model(x)[0]
    prefill, cache = model(x[:, :8])
    rest = []
    for t in range(8, x.size(1)):
        logits, cache = model(x[:, t:t + 1], cache)
        rest.append(logits)
    assert torch.allclose(full, torch.cat([prefill] + rest, 1), atol=1e-4)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_padding_row_stays_zero_after_init(cfg):
    model, _, _ = build_model(cfg)
    assert torch.count_nonzero(model.embed.weight[PAD]) == 0


@pytest.mark.parametrize("cfg", CONFIGS)
def test_init_starts_near_uniform(cfg):
    """An untrained model should sit near ln(vocab). A tied embedding left at
    nn.Embedding's default N(0, 1) saturates the softmax and starts far worse."""
    import math

    import torch.nn as nn

    torch.manual_seed(0)
    model, _, _ = build_model(cfg)
    model.eval()
    x = torch.randint(4, 170, (4, 64))
    y = torch.randint(4, 170, (4, 64))
    logits = model(x)[0]
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    assert loss.item() < math.log(vocab_size()) * 1.5, f"init loss {loss.item():.2f}"
