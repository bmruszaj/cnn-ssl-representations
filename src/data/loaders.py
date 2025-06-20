import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from pathlib import Path
from src.utils.params_yaml import load_yaml

p: dict = load_yaml()
SEED: int = p["seed"]
TR_PERC: float = p["split"]["train_percent"] / 100
TE_PERC: float = p["split"]["test_percent"] / 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / p["paths"]["data_dir"]
SPLIT_FILE = PROJECT_ROOT / p["paths"]["data_dir"] / "cifar_split.npz"

transform: T.Compose = T.Compose(
    [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def get_loaders():
    full_train = CIFAR10(str(DATA_DIR), train=True, download=True, transform=transform)
    full_test = CIFAR10(str(DATA_DIR), train=False, download=True, transform=transform)

    idx = np.load(SPLIT_FILE)
    train_ds = Subset(full_train, idx["train_idx"])
    test_ds = Subset(full_test, idx["test_idx"])

    return (
        DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True),
        DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True),
    )
