"""Convert every MIDI file in a folder to MP3.

    python scripts/to_mp3.py                        # outputs/*.mid -> outputs/mp3/*.mp3
    python scripts/to_mp3.py --soundfont piano.sf2  # much better tone, needs FluidSynth

Needs `pip install lameenc`. FluidSynth is optional: without it the built-in
additive synth is used, which is listenable but obviously synthetic.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pretty_midi  # noqa: E402

HARMONICS = (1.0, 0.45, 0.22, 0.12, 0.06)  # rough piano-ish partial stack
DECAY = 3.2


def _voice(t):
    """Damped harmonic stack. pretty_midi calls this with phase = 2*pi*f*t."""
    tone = sum(a * np.sin(k * t) for k, a in enumerate(HARMONICS, start=1))
    envelope = np.exp(-DECAY * np.linspace(0.0, 1.0, len(t)))
    return tone * envelope / sum(HARMONICS)


def synthesize(pm: pretty_midi.PrettyMIDI, sr: int, soundfont: Optional[str]) -> np.ndarray:
    if soundfont:
        return pm.fluidsynth(fs=sr, sf2_path=soundfont)
    return pm.synthesize(fs=sr, wave=_voice)


def to_pcm16(audio: np.ndarray, headroom: float = 0.89) -> np.ndarray:
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
    enc.set_channels(1)
    enc.set_quality(2)
    return bytes(enc.encode(pcm.tobytes())) + bytes(enc.flush())


def convert(src: Path, dst: Path, sr: int, bitrate: int, soundfont: Optional[str]) -> float:
    pm = pretty_midi.PrettyMIDI(str(src))
    if not any(inst.notes for inst in pm.instruments):
        raise ValueError("no notes")
    audio = synthesize(pm, sr, soundfont)
    dst.write_bytes(encode_mp3(to_pcm16(audio), sr, bitrate))
    return len(audio) / sr


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a folder of MIDI files to MP3.")
    ap.add_argument("--input_dir", default="outputs")
    ap.add_argument("--output_dir", default=None, help="default: <input_dir>/mp3")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--bitrate", type=int, default=192, help="kbps")
    ap.add_argument("--soundfont", default=None, help=".sf2 path; requires FluidSynth")
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
          f"{'  (FluidSynth)' if args.soundfont else '  (built-in synth)'}\n")

    done = failed = skipped = 0
    for src in midis:
        dst = out_dir / (src.stem + ".mp3")
        if dst.exists() and not args.overwrite:
            print(f"  {src.name:24s} skipped (exists; use --overwrite)")
            skipped += 1
            continue
        try:
            seconds = convert(src, dst, args.sr, args.bitrate, args.soundfont)
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
