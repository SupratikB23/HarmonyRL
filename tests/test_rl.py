import torch

from harmonyrl.midi_utils import BOS, EOS, PAD, vocab_size
from harmonyrl.training.rl import (ActorCritic, compute_gae, gather_logprobs,
                                   live_mask, masked_entropy, sequence_rewards)
from harmonyrl.utils import build_model

CFG = {"arch": "lstm", "embed_dim": 32, "hidden": 32, "layers": 1, "dropout": 0.0}


def test_live_mask_stops_at_first_eos():
    t = torch.tensor([[5, 6, EOS, 7, PAD], [5, 6, 7, 8, 9]])
    m = live_mask(t)
    assert m[0].tolist() == [1, 1, 1, 0, 0]
    assert m[1].tolist() == [1, 1, 1, 1, 1]


def test_live_mask_survives_stray_pad():
    """A PAD sampled mid-episode must not truncate the episode: only EOS ends it."""
    t = torch.tensor([[5, PAD, 6, EOS, 9]])
    m = live_mask(t)
    assert m[0].tolist() == [1, 1, 1, 1, 0]
    assert int(m.sum(1).item()) - 1 == 3  # reward lands on the EOS index


def test_gae_credits_the_terminal_reward_backwards():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    adv = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.ones(1, 3))


def test_gae_respects_the_mask():
    rewards = torch.tensor([[0.0, 1.0, 5.0]])
    values = torch.zeros(1, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert adv[0, 2] == 0.0
    assert torch.allclose(adv[0, :2], torch.ones(2))


def test_sequence_rewards_ignores_tokens_after_eos():
    body = [BOS, 20, 120, 150, EOS]
    a, _ = sequence_rewards(torch.tensor([body]), {"diversity": 1.0})
    b, _ = sequence_rewards(torch.tensor([body + [PAD] * 20]), {"diversity": 1.0})
    assert torch.allclose(a, b)


def test_actor_critic_shapes_and_ratio_starts_at_one():
    backbone, _, _ = build_model(CFG)
    model = ActorCritic(backbone).eval()
    x = torch.randint(4, 100, (2, 12))
    logits, values = model(x)
    assert logits.shape == (2, 12, vocab_size())
    assert values.shape == (2, 12)
    lp1 = gather_logprobs(model(x)[0], x)
    lp2 = gather_logprobs(model(x)[0], x)
    assert torch.allclose((lp1 - lp2).exp(), torch.ones_like(lp1))


def test_masked_entropy_ignores_masked_positions():
    logits = torch.zeros(1, 3, vocab_size())
    logits[0, 2] = torch.full((vocab_size(),), -1e9)
    logits[0, 2, 0] = 0.0
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert torch.allclose(masked_entropy(logits, mask),
                          torch.tensor(float(torch.log(torch.tensor(vocab_size() * 1.0)))),
                          atol=1e-4)


def test_perplexity_skips_all_pad_batches():
    from harmonyrl.utils import perplexity
    model, _, _ = build_model(CFG)
    loader = [(torch.randint(4, 100, (2, 8)), torch.randint(4, 100, (2, 8))),
              (torch.full((2, 8), PAD), torch.full((2, 8), PAD))]
    ppl = perplexity(model, loader)
    assert ppl == ppl and ppl > 0  # not NaN
