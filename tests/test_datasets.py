import pretty_midi
import pytest
import torch

from harmonyrl.datasets import MIN_CHUNK_TOKENS, MIDITokenDataset, _group_of, make_loaders

def write_midi(path, n_notes, start_pitch=60):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    for i in range(n_notes):
        t = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=start_pitch + (i % 12),
                                           start=t, end=t + 0.25))
    pm.instruments.append(inst)
    pm.write(str(path))

@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "midi"
    root.mkdir()
    for i in range(6):
        write_midi(root / f"perf_{i}--AUDIO_a_wav--1.midi", 120, 55 + i)
    return root

def test_chunks_cover_more_than_one_sample_per_file(corpus, tmp_path):
    ds = MIDITokenDataset(str(corpus), max_seq_len=64, split="train", train_ratio=1.0,
                          cache_dir=str(tmp_path / "c"))
    assert len(ds) > 6  # the v0.1 pipeline gave exactly one sample per file

def test_short_file_still_yields_a_chunk(tmp_path):
    """A file whose token count lands exactly on MIN_CHUNK_TOKENS is cached, so it
    must also produce a chunk rather than being silently dropped."""
    root = tmp_path / "midi"
    root.mkdir()
    write_midi(root / "a.midi", MIN_CHUNK_TOKENS)
    ds = MIDITokenDataset(str(root), max_seq_len=512, split="train", train_ratio=1.0,
                          cache_dir=str(tmp_path / "c"))
    assert len(ds) >= 1

def test_shapes_and_padding(corpus, tmp_path):
    ds = MIDITokenDataset(str(corpus), max_seq_len=64, split="train", train_ratio=1.0,
                          cache_dir=str(tmp_path / "c"))
    x, y = ds[0]
    assert x.shape == y.shape == (64,)
    assert x.dtype == torch.long
    assert torch.equal(x[1:], y[:-1])  # y is x shifted by one

def test_split_does_not_leak_groups(corpus, tmp_path):
    cache = str(tmp_path / "c")
    tr = MIDITokenDataset(str(corpus), 64, "train", 0.5, cache_dir=cache)
    va = MIDITokenDataset(str(corpus), 64, "val", 0.5, cache_dir=cache)
    tr_spans = {s for s, _ in tr.index}
    va_spans = {s for s, _ in va.index}
    assert not (tr_spans & va_spans)

def test_group_of_collapses_movements():
    a = _group_of("MIDI-Unprocessed_01_R1_2006--AUDIO_01_Track01_wav.midi")
    b = _group_of("MIDI-Unprocessed_01_R1_2006--AUDIO_02_Track02_wav.midi")
    assert a == b

def test_cache_key_tracks_vocab(corpus, tmp_path, monkeypatch):
    import harmonyrl.datasets as D
    cache = tmp_path / "c"
    MIDITokenDataset(str(corpus), 64, "train", 1.0, cache_dir=str(cache))
    before = {p.name for p in cache.iterdir()}
    monkeypatch.setattr(D, "VOCAB_SIZE", 999)
    MIDITokenDataset(str(corpus), 64, "train", 1.0, cache_dir=str(cache))
    assert {p.name for p in cache.iterdir()} != before  # stale cache not reused

def test_make_loaders_batches(corpus, tmp_path):
    tr, va = make_loaders(str(corpus), max_seq_len=64, batch_size=2, train_ratio=0.7,
                          cache_dir=str(tmp_path / "c"))
    X, Y = next(iter(tr))
    assert X.shape == (2, 64) and Y.shape == (2, 64)
