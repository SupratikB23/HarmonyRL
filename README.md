<h1 align="center">🎶 HarmonyRL: Reinforcement Learning for Music Generation 🎶</h1>

<p align="center">
  <b>HarmonyRL</b> is a deep learning framework for generating symbolic music (MIDI) using
  <i>Supervised Learning</i>, <i>Reinforcement Learning (RL)</i>, and <i>Diffusion-based Postprocessing</i>.
</p>

<hr/>

<h2>📌 Project Overview</h2>
<p>
HarmonyRL combines <b>LSTM</b>, <b>Transformer</b> architectures, and <b>Reinforcement Learning</b> (policy gradient–style optimization)
to train generative music models on the <a href="https://magenta.tensorflow.org/datasets/maestro">MAESTRO Dataset</a>. 
The goal is to generate high-quality, coherent, and musically pleasing MIDI outputs.
</p>

<ul>
  <li><b>Supervised Learning:</b> Pretraining with cross-entropy loss on MAESTRO dataset.</li>
  <li><b>Reinforcement Learning:</b> Fine-tuning with reward functions (harmony, rhythm, novelty, smoothness).</li>
  <li><b>Diffusion Postprocessing:</b> Improving raw MIDI outputs by refining transitions and removing dissonance.</li>
</ul>

<hr/>

<h2>⚙️ Tech Stack</h2>

<ul>
  <li><b>Deep Learning:</b> PyTorch (torch>=2.2, torchaudio>=2.2)</li>
  <li><b>Symbolic Music Processing:</b> pretty_midi, mido, music21</li>
  <li><b>Audio Processing:</b> librosa, soundfile</li>
  <li><b>Datasets & Tokenization:</b> HuggingFace datasets>=2.20, transformers>=4.41</li>
  <li><b>Generative Refinement:</b> diffusers>=0.30, accelerate>=0.33</li>
  <li><b>Utilities:</b> numpy, scipy, tqdm, pyyaml</li>
</ul>

<hr/>

<h2>📦 Installation</h2>

```bash
# Clone the repository
git clone https://github.com/yourusername/HarmonyRL.git
cd HarmonyRL

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
```
<hr/> <h2>🎼 Dataset</h2> <p> This project uses the <b>MAESTRO Dataset</b> (approx. 200 hours of virtuosic piano performances, ~1276 MIDI files). Download it from <a href="https://magenta.tensorflow.org/datasets/maestro">Google Magenta MAESTRO</a> and place inside <code>data/maestro/</code>. </p> <hr/> <h2>🚀 Training</h2> <h3>1. Supervised Pretraining</h3>

```bash
python scripts/train_supervised.py --config configs/supervised_config.yaml
```
<h3>2. Reinforcement Learning Fine-tuning</h3>

```bash
python scripts/train_rl.py --config configs/rl_config.yaml
```

<h3>3. Inference (Generate MIDI)</h3>

```bash
python scripts/infer.py --ckpt checkpoints/best_model.pt --output_dir outputs/
```

<hr/> <h2>🧠 Algorithms & Formulations</h2> <h3>1. Supervised Learning</h3> <p> Cross-Entropy Loss is applied on sequence modeling of MIDI tokens: </p> <p align="center"><code>L(θ) = - Σ [ y<sub>t</sub> log P(y<sub>t</sub>|x<sub>&lt;t</sub>; θ) ]</code></p> <h3>2. Reinforcement Learning</h3> <p> We fine-tune the pretrained model using <b>Policy Gradient (REINFORCE)</b> with a baseline to reduce variance: </p> <p align="center"><code>∇J(θ) = E[ (R - b) ∇ log π<sub>θ</sub>(a|s) ]</code></p> <ul> <li>Reward <b>R</b> is computed from multiple components: <ul> <li>Harmony Consistency</li> <li>Rhythmic Structure</li> <li>Novelty & Diversity</li> <li>Smooth Transitions</li> </ul> </li> </ul> <h3>3. Diffusion Postprocessing</h3> <p> Diffusion models (via <code>diffusers</code>) refine generated MIDI embeddings, denoising dissonance and smoothing temporal structure. </p> <hr/> <h2>📊 Configuration</h2> <h3>Supervised Config (configs/supervised_config.yaml)</h3>

```bash
seed: 42
data:
  root: "data/maestro"
  max_seq_len: 2048
  train_ratio: 0.95
model:
  embed_dim: 512
  hidden: 768
  layers: 3
  dropout: 0.2
train:
  batch_size: 8
  lr: 3e-4
  epochs: 20
  clip_grad_norm: 1.0
  ckpt_dir: "checkpoints"
```

<h3>Reinforcement Learning Config (configs/rl_config.yaml)</h3>

```bash
seed: 123
model:
  embed_dim: 512
  hidden: 768
  layers: 3
  dropout: 0.2
train:
  ckpt_dir: "checkpoints"
rl:
  episodes: 2000
  rollout_len: 512
  lr: 1e-5
  baseline_beta: 0.95
  entropy_coef: 0.005
```
<hr/>

<h2>📂 Repository Structure</h2>

<pre>
├── .gitignore
├── README.md
├── config.yaml
├── configs
│   ├── rl_config.yaml
│   └── supervised_config.yaml
├── harmonyrl.egg-info
├── harmonyrl
│   ├── __init__.py
│   ├── datasets.py
│   ├── inference.py
│   ├── midi_utils.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── lstm.py
│   │   └── transformer.py
│   ├── postprocess_diffusers.py
│   ├── rewards.py
│   ├── training
│   │   ├── __init__.py
│   │   ├── rl.py
│   │   └── supervised.py
│   └── utils
│       ├── __init__.py
│       ├── evaluation.py
│       └── logging.py
├── requirements.txt
├── scripts
│   ├── infer.py
│   ├── train_rl.py
│   └── train_supervised.py
└── setup.py
</pre>


<hr/> <h2>📈 Future Improvements</h2> <ul> <li>Experiment with larger Transformer backbones (e.g., Performer, Music Transformer).</li> <li>Introduce <b>Curriculum RL</b> with staged rewards for melody, harmony, and structure.</li> <li>Extend to multi-instrument compositions beyond piano (MAESTRO is piano-only).</li> <li>Integrate <b>GAN-based critics</b> for adversarial refinement of generated music.</li> <li>Better postprocessing via <b>latent diffusion</b> in symbolic space.</li> </ul> <hr/> <h2>🙌 Acknowledgements</h2> <p> - <a href="https://magenta.tensorflow.org/">Magenta Project</a> for MAESTRO dataset.<br/> - PyTorch, HuggingFace, Diffusers team for tools.<br/> - Inspiration from reinforcement learning in sequence generation (REINFORCE, PPO). </p>
