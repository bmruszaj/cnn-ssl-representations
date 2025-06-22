# Self-supervised Pooling Alternatives in ResNet-18
Analyzing classification and representation quality on CIFAR-10 using AE and VAE pooling blocks

---

This repository contains code and experiments for evaluating the impact of replacing the final pooling layer in ResNet-18 with self-supervised pooling modules (Autoencoder, Variational Autoencoder, SimCLR). The goal is to compare classification performance and representation quality on the CIFAR-10 dataset.

---

## 📁 Project structure

```
.
├── data/                          # CIFAR-10 dataset and train/test splits
├── src/
│   ├── data/                     # data loaders and split preparation
│   ├── models/                   # model architectures (AE, VAE, ResNet)
│   └── utils/                    # helper functions (YAML parameter loader)
├── pretrain/                     # scripts to pre-train AE and VAE pooling blocks
├── experiments/                  # training, evaluation, and visualization scripts
├── results/                      # saved metrics, models, and plots
├── params.yaml                   # hyperparameters and paths
├── dvc.yaml / dvc.lock           # DVC pipeline definitions and lock file
├── Makefile                      # convenience commands (install, lint, docker)
├── cpu.dockerfile / gpu.dockerfile
├── requirements-cpu.txt / requirements-gpu.txt
└── README.md                     # this file
```

---

## ⚙️ Environment Setup

Requirements: Python 3.12, pip ≥ 23. For GPU support, CUDA-compatible drivers are needed.

1. Clone the repository and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
   ```

2. Install dependencies for CPU or GPU:
   ```bash
   make install DEVICE=cpu
   # or
   make install DEVICE=gpu
   ```

3. (Optional) Launch Jupyter Lab in Docker:
   ```bash
   make run_container DEVICE=cpu
   ```

---

## 🐳 Docker Usage

- Build image:
  ```bash
  make build_image DEVICE=cpu  # or DEVICE=gpu
  ```
- Run container:
  ```bash
  make run_container DEVICE=gpu
  ```

---

## 🧪 Experiments

- `src/data/prepare_split.py` – generate train/test indices (DVC stage: prepare_split)
- `pretrain/pretrain_pool_encoder.py` – pre-train AE and VAE pooling blocks (DVC stage: pretrain_pool)
- `experiments/train_baseline.py` – train standard ResNet-18 classifier (DVC: train_baseline)
- `experiments/train_resnet_with_pool.py` – train ResNet-18 using pretrained AE/VAE pooling (frozen or unfrozen) (DVC: train_with_pool)
- `experiments/evaluate.py` – evaluate model accuracy (DVC: evaluate)
- `experiments/summarize.py` – aggregate and plot results (DVC: summarize)
- `experiments/show_photo_matrix.py` – visualize learned representations

Notebooks for further analysis:
- `experiments/simclr_experiments.ipynb`

To run the full DVC pipeline:
```bash
 dvc repro
```

---

## 📦 System Requirements

- Python 3.12
- pip ≥ 23
- (GPU only) CUDA 12.1+
- Docker (optional)

---

## 🔄 Reproducibility

All hyperparameters and paths are defined in `params.yaml`. To reproduce results:
```bash
 dvc repro
```

---

## 💾 DVC Remote Configuration

This project uses a remote DVC storage backend. To access and push data to the remote, create a `config.local` file in the repository root (this file is git-ignored) with your remote storage credentials. For example:

```ini
['remote "gdrive_remote"']
gdrive_client_id = YOUR_ACCESS_KEY_ID
gdrive_client_secret = YOUR_SECRET_ACCESS_KEY
```

You need to replace values with the ones that creators share. Once saved, you can run `dvc pull` / `dvc push` to sync data.
