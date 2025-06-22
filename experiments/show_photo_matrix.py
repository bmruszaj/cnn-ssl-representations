import random
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

ROOT       = Path("../data")
OUT_DIR    = Path("../results/plt")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE   = OUT_DIR / "cifar10_grid.png"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog",     "frog",       "horse", "ship", "truck",
]
SAMPLES_PER_CLASS = 10

ds = CIFAR10(root=str(ROOT), train=True, download=True, transform=T.ToTensor())

idx_by_class = {c: [] for c in range(10)}
for idx, (_, label) in enumerate(ds):
    if len(idx_by_class[label]) < SAMPLES_PER_CLASS:
        idx_by_class[label].append(idx)
    if all(len(lst) == SAMPLES_PER_CLASS for lst in idx_by_class.values()):
        break

fig, axes = plt.subplots(
    nrows=len(CLASSES),
    ncols=SAMPLES_PER_CLASS + 1,
    figsize=((SAMPLES_PER_CLASS + 1) * 1.6, len(CLASSES) * 1.6),
)
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for row, class_name in enumerate(CLASSES):
    axes[row, 0].text(
        0.0, 0.5, class_name,
        fontsize=10, fontweight="bold",
        va="center", ha="left", transform=axes[row, 0].transAxes,
    )
    axes[row, 0].axis("off")

    for col, ds_idx in enumerate(idx_by_class[row], start=1):
        img, _ = ds[ds_idx]                     # tensor [3×32×32], already 0–1
        axes[row, col].imshow(img.permute(1, 2, 0))
        axes[row, col].axis("off")

fig.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print("✓ zapisano:", OUT_FILE)
