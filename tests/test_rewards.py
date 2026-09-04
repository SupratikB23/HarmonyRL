import random

from harmonyrl import midi_utils as mu
from harmonyrl.rewards import reward_parts, total_reward
from harmonyrl.utils.evaluation import max_repeat_run

WEIGHTS = {"harmony": 0.3, "scale": 0.2, "rhythm": 0.2, "diversity": 0.2, "density": 0.1}


def note(pitch, dur):
    return [mu.PITCH_BASE + pitch - mu.MIN_PITCH, mu.VEL_BASE + 20, mu.DUR_BASE + dur - 1]


def seq(notes):
    return [mu.BOS, mu.BAR] + sum((note(p, d) for p, d in notes), []) + [mu.EOS]


def test_repeated_note_is_not_optimal():
    random.seed(0)
    loop = seq([(60, 4)] * 40)
    tonal = seq([(60 + random.choice([0, 2, 4, 5, 7, 9, 11]), random.choice([2, 4, 8]))
                 for _ in range(40)])
    assert total_reward(loop, WEIGHTS) < total_reward(tonal, WEIGHTS)
    assert reward_parts(loop)["diversity"] < 0.1


def test_all_parts_bounded():
    random.seed(1)
    s = seq([(random.randint(21, 108), random.choice([1, 2, 4])) for _ in range(30)])
    for name, v in reward_parts(s).items():
        assert -1.0 <= v <= 1.0, name


def test_empty_sequence_is_zero():
    assert total_reward([mu.BOS, mu.EOS], WEIGHTS) == 0.0


def test_max_repeat_run():
    assert max_repeat_run(seq([(60, 4)] * 5)) == 5
    assert max_repeat_run(seq([(60, 4), (62, 4), (64, 4)])) == 1
