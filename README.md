<p align="center">
  <h1 align="center">🌊 S E I S M O G E N</h1>
  <p align="center">
    <em>A Hybrid Mamba–U-Net Diffusion Framework for Conditional Earthquake Waveform Synthesis</em>
  </p>
  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#key-features">Key Features</a> •
    <a href="#architecture--tech-stack">Architecture</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#usage">Usage</a> •
    <a href="#project-structure">Project Structure</a> •
    <a href="#license">License</a>
  </p>
</p>

---

## Overview

**SeismoGen** is a deep generative framework that synthesizes realistic, physically-conditioned earthquake seismograms from pure noise. It combines the long-range sequential modeling power of **Mamba State Space Models** with the hierarchical feature extraction of **1D U-Net architectures**, all driven by a **Denoising Diffusion Probabilistic Model (DDPM)** scheduler.

The system takes a 7-dimensional seismic metadata vector (e.g., magnitude, depth, epicentral distance, station parameters) as a conditioning signal and iteratively denoises Gaussian noise into a 3-channel seismogram representing the East (E), North (N), and Vertical (Z) components of ground motion.

**Why does this exist?** Real earthquake waveform datasets are sparse, imbalanced, and expensive to acquire. SeismoGen enables researchers to generate unlimited synthetic seismograms conditioned on arbitrary source parameters — augmenting training data for earthquake detection models, stress-testing early warning systems, and exploring seismic hazard scenarios that have never been observed.

---

## Key Features

- **Bidirectional Mamba Bottleneck** — Processes temporal sequences in both forward and reverse directions simultaneously, eliminating causal bias that would distort denoising of seismic signals where P-waves and S-waves propagate in both temporal directions.
- **Hybrid U-Net + SSM Architecture** — Combines the local feature extraction of 1D convolutional ResBlocks with the O(n) long-range sequence modeling of Mamba, avoiding the quadratic cost of Transformer attention.
- **DDPM Noise Scheduling** — Full denoising diffusion pipeline with configurable timesteps (default: 1000), supporting both training (forward noise addition) and generation (reverse iterative denoising).
- **Conditional Generation** — Metadata-conditioned synthesis via fused sinusoidal timestep embeddings and learned condition embeddings, enabling fine-grained control over the generated waveform characteristics.
- **Adaptive FiLM Conditioning** — Feature-wise Linear Modulation (scale + shift) in every ResBlock, allowing the conditioning signal to modulate features at every level of the network hierarchy.
- **Multi-Channel Output** — Generates all three seismogram components (E/N/Z) simultaneously as a 3-channel 1D signal.
- **Variable Sequence Length** — Supports arbitrary waveform lengths (128, 256, 512, …) with automatic padding alignment in skip connections.

---

## Architecture & Tech Stack

### Model Architecture

```
Input (Noisy Waveform)          Conditioning Vector (7-D metadata)
       │                                  │
       ▼                                  ▼
  ┌─────────┐                    ┌──────────────────┐
  │ Conv1D  │                    │ ConditionEmbed    │
  │  Stem   │                    │ (MLP)             │
  └────┬────┘                    └────────┬─────────┘
       │                                  │
       │         Timestep ────► SinusoidalEmbed ──┐
       │                                          │
       │                              ┌───────────┴──┐
       │                              │  Fused Emb   │
       │                              │  (t + cond)  │
       │                              └───────┬──────┘
       │                                      │
  ┌────┼──────────────────────────────────────┼────┐
  │    ▼              ENCODER                 │    │
  │ ResBlock1D(64)  ◄── FiLM ────────────────►│    │
  │    │  skip₁                                    │
  │    ▼                                           │
  │ ↓ Downsample (stride-2 Conv1D)                 │
  │    ▼                                           │
  │ ResBlock1D(128) ◄── FiLM ────────────────►│    │
  │    │  skip₂                                    │
  │    ▼                                           │
  │ ↓ Downsample (stride-2 Conv1D)                 │
  └────┼───────────────────────────────────────────┘
       │
  ┌────▼───────────────────────────────────────────┐
  │              BOTTLENECK                        │
  │  ResBlock1D(128) ◄── FiLM                     │
  │       │                                        │
  │       ▼                                        │
  │  ┌──────────────────────────┐                  │
  │  │  BiMamba (Bidirectional) │                  │
  │  │  ├─ Forward Mamba Scan   │                  │
  │  │  ├─ Backward Mamba Scan  │                  │
  │  │  └─ LayerNorm + Residual │                  │
  │  └──────────────────────────┘                  │
  │       │                                        │
  │  ResBlock1D(128) ◄── FiLM                     │
  └────┬───────────────────────────────────────────┘
       │
  ┌────▼───────────────────────────────────────────┐
  │              DECODER                           │
  │ ↑ Upsample (nearest) + concat(skip₂)          │
  │ ResBlock1D(256→64) ◄── FiLM                   │
  │    │                                           │
  │ ↑ Upsample (nearest) + concat(skip₁)          │
  │ ResBlock1D(128→64) ◄── FiLM                   │
  └────┬───────────────────────────────────────────┘
       │
       ▼
  GroupNorm → SiLU → Conv1D(64 → 3)
       │
       ▼
  Predicted Noise (ε̂)
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12 |
| **Deep Learning** | PyTorch |
| **State Space Model** | [Mamba](https://github.com/state-spaces/mamba) (`mamba-ssm`) |
| **Diffusion Scheduler** | Custom DDPM implementation |
| **Training** | Adam optimizer, MSE loss (noise prediction objective) |
| **Hardware** | CUDA-accelerated (auto-fallback to CPU) |
| **Experiment Tracking** | Weights & Biases (`wandb`) |
| **Data** | HuggingFace `datasets`, Kaggle API |

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.10
- **CUDA** ≥ 11.8 (required for `mamba-ssm`)
- **Git**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/seismogen.git
   cd seismogen
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate        # Linux/macOS
   myenv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install mamba-ssm causal-conv1d
   pip install -r requirements.txt
   ```

> [!IMPORTANT]
> `mamba-ssm` requires a CUDA-capable GPU and a working CUDA toolkit. See the [Mamba installation guide](https://github.com/state-spaces/mamba#installation) if you encounter build issues.

---

## Usage

### Training

Prepare your seismic dataset as a PyTorch `DataLoader` that yields `(waveform, condition)` tuples, where:
- `waveform` has shape `(batch, 3, seq_length)` — the 3-channel seismogram
- `condition` has shape `(batch, 7)` — the metadata vector

Then run:

```bash
python train.py
```

The training loop will:
1. Add noise to clean waveforms using the DDPM forward process
2. Predict the added noise using the model
3. Optimize with MSE loss
4. Evaluate on validation data each epoch
5. Save weights to `mamba_diffusion_weights.pth`


## License

This project is released for academic and research purposes. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with 🔬 PyTorch • 🐍 Mamba SSM • 🌍 Seismology</sub>
</p>

