from typing import List
import numpy as np
import pretty_midi

# --- very compact tokenization scheme ---
SPECIAL = {"PAD":0, "BOS":1, "EOS":2, "BAR":3, "REST":4}
PITCH_OFFSET = 16
MIN_PITCH, MAX_PITCH = 21, 108
N_PITCH = MAX_PITCH - MIN_PITCH + 1
DUR_BASE = 256
DURS = [60, 120, 240, 480, 960]  # ticks @ 480PPQ
VOCAB_SIZE = DUR_BASE + len(DURS)

def vocab_size() -> int:
    return VOCAB_SIZE

def pitch_to_token(p: int) -> int:
    if p < MIN_PITCH or p > MAX_PITCH: return SPECIAL["REST"]
    return PITCH_OFFSET + (p - MIN_PITCH)

def token_is_pitch(tok: int) -> bool:
    return PITCH_OFFSET <= tok < PITCH_OFFSET + N_PITCH

def token_to_pitch(tok: int) -> int:
    return MIN_PITCH + (tok - PITCH_OFFSET)

def dur_to_token(d: int) -> int:
    idx = int(np.argmin([abs(d - x) for x in DURS]))
    return DUR_BASE + idx

def token_to_dur(tok: int) -> int:
    return DURS[tok - DUR_BASE]

def midi_to_tokens(pm: pretty_midi.PrettyMIDI, max_len=2048) -> List[int]:
    notes = []
    for inst in pm.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    notes.sort(key=lambda x: x[0])

    seq = [SPECIAL["BOS"]]
    last_end = 0.0
    for s, e, p in notes:
        if len(seq) >= max_len-3: break
        if s > last_end:
            seq += [SPECIAL["REST"], dur_to_token(int((s - last_end) * 480))]
        seq += [pitch_to_token(p), dur_to_token(int((e - s) * 480))]
        last_end = e
        if int(e*2) % 2 == 0: seq.append(SPECIAL["BAR"])
    seq.append(SPECIAL["EOS"])
    return seq[:max_len]

def tokens_to_midi(tokens: List[int], tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if token_is_pitch(tok):
            dur = 0.5
            if i+1 < len(tokens) and tokens[i+1] >= DUR_BASE:
                dur = token_to_dur(tokens[i+1]) / 480.0
                i += 1
            p = token_to_pitch(tok)
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=t, end=t+dur))
            t += dur
        elif tok == SPECIAL["REST"]:
            if i+1 < len(tokens) and tokens[i+1] >= DUR_BASE:
                t += token_to_dur(tokens[i+1]) / 480.0
                i += 1
        elif tok == SPECIAL["EOS"]:
            break
        i += 1
    pm.instruments.append(inst)
    return pm

def synth_audio(pm: pretty_midi.PrettyMIDI, sr: int = 32000):
    # pretty_midi's synth (FluidSynth if installed)
    try:
        return pm.fluidsynth(fs=sr)
    except Exception:
        return pm.synthesize(fs=sr)
