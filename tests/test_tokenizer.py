import pretty_midi
import pytest

from harmonyrl import midi_utils as mu


def make_midi():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    for start, pitches in [(0.0, [60, 64, 67]), (0.5, [62]), (1.0, [65, 69])]:
        for p in pitches:
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=start, end=start + 0.5))
    pm.instruments.append(inst)
    return pm


def test_vocab_is_dense():
    ids = set()
    for lo, hi in [(0, 4), (mu.POS_BASE, mu.PITCH_BASE), (mu.PITCH_BASE, mu.VEL_BASE),
                   (mu.VEL_BASE, mu.DUR_BASE), (mu.DUR_BASE, mu.VOCAB_SIZE)]:
        ids.update(range(lo, hi))
    assert ids == set(range(mu.VOCAB_SIZE))


def test_roundtrip_preserves_notes_and_polyphony():
    pm = make_midi()
    tokens = mu.midi_to_tokens(pm, 4096)
    out = mu.tokens_to_midi(tokens).instruments[0].notes
    assert len(out) == 6
    assert sorted(n.pitch for n in out) == [60, 62, 64, 65, 67, 69]
    starts = [round(n.start, 3) for n in out]
    assert starts.count(min(starts)) == 3  # the opening triad stays simultaneous


def test_max_len_is_respected():
    assert len(mu.midi_to_tokens(make_midi(), 12)) <= 12


def test_empty_midi():
    assert mu.midi_to_tokens(pretty_midi.PrettyMIDI()) == [mu.BOS, mu.EOS]


@pytest.mark.parametrize("tokens", [[mu.BOS, mu.EOS], [mu.PITCH_BASE], [mu.DUR_BASE, mu.BAR, 5]])
def test_decoder_tolerates_malformed(tokens):
    mu.tokens_to_midi(tokens)


def test_decode_has_no_leading_silence():
    """The first BAR must not advance the clock: bar 0 starts at t=0."""
    tokens = [mu.BOS, mu.BAR, mu.POS_BASE, mu.PITCH_BASE + 39,
              mu.VEL_BASE + 20, mu.DUR_BASE + 3, mu.EOS]
    notes = mu.tokens_to_midi(tokens).instruments[0].notes
    assert len(notes) == 1
    assert notes[0].start == 0.0


def test_roundtrip_start_time_is_preserved():
    pm = make_midi()
    out = mu.tokens_to_midi(mu.midi_to_tokens(pm, 4096)).instruments[0].notes
    assert min(n.start for n in out) == 0.0
