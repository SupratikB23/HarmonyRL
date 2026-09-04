from typing import List, Sequence

import pretty_midi

# REMI-style event vocabulary: BAR, Position, Pitch, Velocity, Duration.
# Contiguous ids, no reserved gaps -- every id is reachable.
STEPS_PER_BEAT = 4
STEPS_PER_BAR = 16
MIN_PITCH, MAX_PITCH = 21, 108
N_PITCH = MAX_PITCH - MIN_PITCH + 1
N_VELOCITY = 32
MAX_DURATION = 32
MAX_EMPTY_BARS = 2

PAD, BOS, EOS, BAR = 0, 1, 2, 3
POS_BASE = 4
PITCH_BASE = POS_BASE + STEPS_PER_BAR
VEL_BASE = PITCH_BASE + N_PITCH
DUR_BASE = VEL_BASE + N_VELOCITY
VOCAB_SIZE = DUR_BASE + MAX_DURATION

TOKENS_PER_NOTE = 4


def vocab_size() -> int:
    return VOCAB_SIZE


def is_position(tok: int) -> bool:
    return POS_BASE <= tok < PITCH_BASE


def is_pitch(tok: int) -> bool:
    return PITCH_BASE <= tok < VEL_BASE


def is_velocity(tok: int) -> bool:
    return VEL_BASE <= tok < DUR_BASE


def is_duration(tok: int) -> bool:
    return DUR_BASE <= tok < VOCAB_SIZE


def token_to_position(tok: int) -> int:
    return tok - POS_BASE


def token_to_pitch(tok: int) -> int:
    return tok - PITCH_BASE + MIN_PITCH


def token_to_velocity(tok: int) -> int:
    return min(127, (tok - VEL_BASE) * (128 // N_VELOCITY) + 4)


def token_to_duration(tok: int) -> int:
    return tok - DUR_BASE + 1


def _note_events(pm: pretty_midi.PrettyMIDI):
    ticks_per_step = max(1, round(pm.resolution / STEPS_PER_BEAT))
    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if not MIN_PITCH <= n.pitch <= MAX_PITCH:
                continue
            start = round(pm.time_to_tick(n.start) / ticks_per_step)
            end = round(pm.time_to_tick(n.end) / ticks_per_step)
            dur = min(MAX_DURATION, max(1, end - start))
            events.append((start, n.pitch, n.velocity, dur))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def midi_to_tokens(pm: pretty_midi.PrettyMIDI, max_len: int = 2048) -> List[int]:
    events = _note_events(pm)
    if not events:
        return [BOS, EOS]

    origin = events[0][0] - events[0][0] % STEPS_PER_BAR
    seq = [BOS]
    cur_bar = -1
    for step, pitch, vel, dur in events:
        if len(seq) + TOKENS_PER_NOTE + MAX_EMPTY_BARS >= max_len:
            break
        bar, pos = divmod(step - origin, STEPS_PER_BAR)
        if bar != cur_bar:
            seq += [BAR] * (1 if cur_bar < 0 else min(bar - cur_bar, MAX_EMPTY_BARS))
            cur_bar = bar
        seq += [
            POS_BASE + pos,
            PITCH_BASE + pitch - MIN_PITCH,
            VEL_BASE + min(N_VELOCITY - 1, vel * N_VELOCITY // 128),
            DUR_BASE + dur - 1,
        ]
    seq.append(EOS)
    return seq


def tokens_to_midi(tokens: Sequence[int], tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """Decode tolerantly -- generated sequences are often malformed."""
    sec_per_step = 60.0 / tempo / STEPS_PER_BEAT
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)

    # The encoder emits a BAR before the first note of bar 0, so the first BAR
    # token must land on bar 0 rather than push everything one bar later.
    bar, pos, seen_bar = 0, 0, False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == EOS:
            break
        if tok == BAR:
            bar = bar + 1 if seen_bar else 0
            seen_bar = True
            pos = 0
        elif is_position(tok):
            pos = token_to_position(tok)
        elif is_pitch(tok):
            vel, dur = 80, STEPS_PER_BEAT
            if i + 1 < len(tokens) and is_velocity(tokens[i + 1]):
                vel = token_to_velocity(tokens[i + 1])
                i += 1
            if i + 1 < len(tokens) and is_duration(tokens[i + 1]):
                dur = token_to_duration(tokens[i + 1])
                i += 1
            start = (bar * STEPS_PER_BAR + pos) * sec_per_step
            inst.notes.append(
                pretty_midi.Note(velocity=vel, pitch=token_to_pitch(tok),
                                 start=start, end=start + dur * sec_per_step)
            )
        i += 1

    pm.instruments.append(inst)
    return pm


def synth_audio(pm: pretty_midi.PrettyMIDI, sr: int = 32000):
    try:
        return pm.fluidsynth(fs=sr)
    except Exception:
        return pm.synthesize(fs=sr)
