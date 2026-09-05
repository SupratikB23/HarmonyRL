"""Convert every MIDI file in a folder to MP3.

    python scripts/to_mp3.py                        # outputs/*.mid -> outputs/mp3/*.mp3
    python scripts/to_mp3.py --max_note 1.5         # tame long-note pileups
    python scripts/to_mp3.py --soundfont piano.sf2  # real samples, needs FluidSynth

Needs `pip install lameenc`. Without a soundfont it uses the built-in additive
piano below: inharmonic partials, real-time decay, stereo, and a small room.
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pretty_midi  # noqa: E402

N_PARTIALS = 10
INHARMONICITY = 4e-4  # real piano strings are stiff, so partials run sharp
RELEASE = 0.35        # ring-out past note-off
DAMPER = 12.0         # how fast the damper kills a released note


def _note_wave(freq: float, dur: float, vel: int, sr: int, rng) -> np.ndarray:
    """One piano-ish note: stiff-string partials, each decaying at its own rate."""
    n = int((dur + RELEASE) * sr)
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n, dtype=np.float64) / sr
    v = vel / 127.0

    # Loud notes are brighter: a lower exponent keeps more energy up high.
    tilt = 2.9 - 0.8 * v
    # Bass strings ring far longer than treble.
    base_decay = 0.7 + 9.0 * (freq / 4186.0) ** 0.55

    out = np.zeros(n)
    for k in range(1, N_PARTIALS + 1):
        fk = freq * k * math.sqrt(1.0 + INHARMONICITY * k * k)
        if fk >= 0.45 * sr:  # stay well clear of Nyquist
            break
        decay = base_decay * (1.0 + 0.45 * (k - 1))  # highs die first
        phase = rng.uniform(0.0, 2.0 * math.pi)      # no stacked click on chords
        out += (1.0 / k ** tilt) * np.sin(2.0 * math.pi * fk * t + phase) * np.exp(-decay * t)

    attack = max(1, int(0.005 * sr))
    out[:attack] *= np.linspace(0.0, 1.0, attack)

    off = int(dur * sr)
    if off < n:  # damper drops on note-off
        out[off:] *= np.exp(-DAMPER * (t[off:] - t[off]))

    return out * (v ** 1.5)


def _reverb(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Small room, as a decaying-noise convolution done on the FFT."""
    if amount <= 0:
        return audio
    rng = np.random.default_rng(0)
    n_ir = int(0.7 * sr)
    t = np.arange(n_ir) / sr
    ir = rng.standard_normal(n_ir) * np.exp(-5.5 * t)
    ir[0] = 1.0
    ir /= np.abs(ir).sum()

    fft_len = 1 << (len(audio) + n_ir - 2).bit_length()
    wet = np.fft.irfft(np.fft.rfft(audio, fft_len) * np.fft.rfft(ir, fft_len))[: len(audio)]
    return (1.0 - amount) * audio + amount * wet


def _tone_shape(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """One-pole rolloff above `cutoff`, applied on the FFT. A bare partial stack is
    far brighter than a recorded piano; this puts the spectral tilt back."""
    n = 1 << (len(audio) - 1).bit_length()
    spec = np.fft.rfft(audio, n)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(spec / np.sqrt(1.0 + (f / cutoff) ** 2))[: len(audio)]


def render(pm: pretty_midi.PrettyMIDI, sr: int, max_note: Optional[float],
           reverb: float, brightness: float) -> np.ndarray:
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        raise ValueError("no notes")

    end = max(n.end for n in notes) + RELEASE + 0.2
    left = np.zeros(int(end * sr) + 1)
    right = np.zeros_like(left)
    rng = np.random.default_rng(0)

    for note in notes:
        dur = note.end - note.start
        if max_note:
            dur = min(dur, max_note)
        wave = _note_wave(440.0 * 2.0 ** ((note.pitch - 69) / 12.0), dur,
                          note.velocity, sr, rng)
        i = int(note.start * sr)
        j = min(i + len(wave), len(left))
        if j <= i:
            continue
        seg = wave[: j - i]
        # gentle stereo spread: low notes left, high notes right
        pan = np.clip((note.pitch - 60) / 48.0, -1.0, 1.0) * 0.35
        left[i:j] += seg * math.sqrt(0.5 * (1.0 - pan))
        right[i:j] += seg * math.sqrt(0.5 * (1.0 + pan))

    left = _tone_shape(left, sr, brightness)
    right = _tone_shape(right, sr, brightness)
    stereo = np.stack([_reverb(left, sr, reverb), _reverb(right, sr, reverb)], axis=1)

    # Normalize on a high percentile rather than the peak, so one dense chord
    # cannot crush the whole piece; tanh then catches what overshoots.
    ref = np.percentile(np.abs(stereo), 99.5)
    if ref > 0:
        stereo = np.tanh(stereo / ref * 0.85)
    return stereo


def synthesize(pm, sr, soundfont, max_note, reverb, brightness) -> np.ndarray:
    if soundfont:
        mono = pm.fluidsynth(fs=sr, sf2_path=soundfont)
        return np.stack([mono, mono], axis=1)
    return render(pm, sr, max_note, reverb, brightness)


def to_pcm16(audio: np.ndarray, headroom: float = 0.80) -> np.ndarray:
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float64))
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * headroom
    return (audio * 32767.0).astype(np.int16)


def encode_mp3(pcm: np.ndarray, sr: int, bitrate: int) -> bytes:
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(2 if pcm.ndim == 2 else 1)
    enc.set_quality(2)
    return bytes(enc.encode(pcm.tobytes())) + bytes(enc.flush())


def convert(src: Path, dst: Path, sr: int, bitrate: int, soundfont: Optional[str],
            max_note: Optional[float], reverb: float, brightness: float) -> float:
    pm = pretty_midi.PrettyMIDI(str(src))
    audio = synthesize(pm, sr, soundfont, max_note, reverb, brightness)
    dst.write_bytes(encode_mp3(to_pcm16(audio), sr, bitrate))
    return len(audio) / sr


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a folder of MIDI files to MP3.")
    ap.add_argument("--input_dir", default="outputs")
    ap.add_argument("--output_dir", default=None, help="default: <input_dir>/mp3")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--bitrate", type=int, default=192, help="kbps")
    ap.add_argument("--soundfont", default=None, help=".sf2 path; requires FluidSynth")
    ap.add_argument("--max_note", type=float, default=None,
                    help="cap note length in seconds; thins long-note pileups")
    ap.add_argument("--reverb", type=float, default=0.22, help="0 disables")
    ap.add_argument("--brightness", type=float, default=1400.0,
                    help="tone rolloff in Hz; raise for a brighter piano")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if importlib.util.find_spec("lameenc") is None:
        print("Missing encoder. Run:  pip install lameenc", file=sys.stderr)
        return 1

    src_dir = Path(args.input_dir)
    if not src_dir.is_dir():
        print(f"No such folder: {src_dir}", file=sys.stderr)
        return 1

    midis = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in (".mid", ".midi"))
    if not midis:
        print(f"No MIDI files in {src_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else src_dir / "mp3"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(midis)} MIDI -> {out_dir}"
          f"{'  (FluidSynth)' if args.soundfont else '  (built-in piano)'}\n")

    done = failed = skipped = 0
    for src in midis:
        dst = out_dir / (src.stem + ".mp3")
        if dst.exists() and not args.overwrite:
            print(f"  {src.name:24s} skipped (exists; use --overwrite)")
            skipped += 1
            continue
        try:
            seconds = convert(src, dst, args.sr, args.bitrate, args.soundfont,
                              args.max_note, args.reverb, args.brightness)
        except Exception as e:
            print(f"  {src.name:24s} FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        print(f"  {src.name:24s} -> {dst.name:24s} {seconds:6.1f}s  "
              f"{dst.stat().st_size / 1024:7.1f} KB")
        done += 1

    print(f"\n{done} converted, {skipped} skipped, {failed} failed -> {out_dir}")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
