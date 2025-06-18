# Self-supervised alternatives to pooling in VGG-19  
Analyzing classification and representation quality on CIFAR-10

---

This repository contains code and experiments for evaluating the impact of replacing the final max-pooling layer in the VGG-19 architecture with autoencoders, SimCLR, and BYOL.  
The goal is to compare both the classification performance and the quality of learned representations on the CIFAR-10 dataset.  
The project is developed as part of the **Representation Learning (2025)** course.

---

## 📁 Project structure

```
.
├── data/           # raw dataset or CIFAR-10 loader
├── src/            # model architectures, loss functions, utilities
├── experiments/    # scripts to train AE / SimCLR / BYOL
├── results/        # saved metrics, models, and plots
├── notebooks/      # Jupyter notebooks for result analysis
├── params.yaml     # hyperparameters (for DVC)
├── dvc.yaml        # pipeline definition (optional, if using DVC)
├── requirements.txt
├── Makefile
└── README.md
```

---

## ⚙️ Environment setup

The project can be run locally or in a Docker container.  
Code is compatible with both CPU and GPU. Dependency installation depends on the device used — simply specify `DEVICE=cpu` or `DEVICE=gpu`.

---

### 🔧 Local installation (Python 3.12)

We recommend using `virtualenv` or `conda`.

1. Create and activate a virtual environment.

2. Install dependencies:

```bash
make install DEVICE=cpu
# or
make install DEVICE=gpu
```

3. Launch Jupyter Notebook or Lab:

```bash
jupyter notebook
# or
jupyter lab
```

---

### 🐳 Docker installation

1. Build the image:

```bash
make build_image DEVICE=cpu
# or
make build_image DEVICE=gpu
```

2. Run the container (default port is 8888; you can override it with `PORT=8080`):

```bash
make run_container DEVICE=cpu
```

A link with a token will be displayed in the terminal for accessing the Jupyter interface.

---

## 📦 System requirements

- Python 3.12
- pip ≥ 23
- (for GPU) CUDA-compatible GPU (CUDA 12.1+)
- Docker (if using containers)

---

## 🧪 Experiments

- `experiments/baseline.py` – standard VGG-19
- `experiments/train_ae.py` – VGG + autoencoder
- `experiments/train_simclr.py` – SimCLR as pooling replacement
- `experiments/train_byol.py` – BYOL as pooling replacement
- `experiments/evaluate.py` – embedding quality evaluation (UMAP, CKA, RankMe)

---

## 📊 Reproducibility

- All hyperparameters are defined in `params.yaml`
- The experiment pipeline is described in `dvc.yaml`
- To reproduce results:

```bash
dvc repro
```

---

## 👥 Team

- Person A – AE/VAE
- Person B – SimCLR
- Person C – BYOL + loss modification

---

Need help with setup or training? Feel free to reach out! 💡
