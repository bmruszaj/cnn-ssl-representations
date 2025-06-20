import numpy as np
from pathlib import Path
from torchvision.datasets import CIFAR10
from src.utils.params_yaml import load_yaml

p = load_yaml()
root = Path(__file__).resolve().parent.parent.parent
data_dir = root / p["paths"]["data_dir"]
out_file = root / p["paths"]["data_dir"] / "cifar_split.npz"
out_file.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.RandomState(p["seed"])
train_len = len(CIFAR10(str(data_dir), train=True, download=True))
test_len = len(CIFAR10(str(data_dir), train=False, download=True))

tr_size = int(train_len * p["split"]["train_percent"] / 100)
te_size = int(test_len * p["split"]["test_percent"] / 100)

train_idx = rng.choice(train_len, tr_size, replace=False)
test_idx = rng.choice(test_len, te_size, replace=False)

np.savez(out_file, train_idx=train_idx, test_idx=test_idx)
print("Split saved to", out_file)
