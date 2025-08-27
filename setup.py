from setuptools import setup, find_packages

setup(
    name="harmonyrl",
    version="0.1.0",
    description="HarmonyRL: Multi-instrument music generation with LSTM/Transformer + RL fine-tuning",
    packages=find_packages(exclude=("tests", "notebooks")),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2",
        "torchaudio>=2.2",
        "pretty_midi",
        "mido",
        "numpy",
        "scipy",
        "tqdm",
        "pyyaml",
        "music21",
        "soundfile",
        "librosa",
        "transformers>=4.41",
        "datasets>=2.20",
        "diffusers>=0.30",
        "accelerate>=0.33",
    ],
)