# experiments/eval_model.py
import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any, Literal

from src.utils.params_yaml import load_yaml
from src.models.resnet18_with_pools import build_resnet18
from src.data.loaders import get_loaders

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
params: Dict[str, Any] = load_yaml()
ALLOWED: tuple[Literal["baseline", "ae", "vae", "simclr", "byol"], ...] = (
    "baseline",
    "ae",
    "vae",
    "simclr",
    "byol",
)


# ----------------------------------------------------------------------------- #
# Funkcje                                                                        #
# ----------------------------------------------------------------------------- #
def evaluate(model_type: str) -> None:
    """Załaduj checkpoint i policz accuracy dla wybranego wariantu."""
    if model_type not in ALLOWED:
        raise ValueError(f"model_type must be one of {ALLOWED}")

    paths = params["paths"]
    ckpt_path = PROJECT_ROOT / paths["checkpoints_dir"] / f"{model_type}_model.pt"
    results_path = PROJECT_ROOT / paths["results_dir"] / f"{model_type}_acc.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders()

    model = build_resnet18(
        pretrained=False,
        pool_type=model_type if model_type != "baseline" else "max",
        freeze_pool=False,
        latent=params["model"]["latent_channels"],
        num_classes=params["model"]["num_classes"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Accuracy ({model_type}): {acc:.2f}%")

    with open(results_path, "w") as f:
        json.dump({"accuracy": acc}, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint.")
    parser.add_argument(
        "--model_type",
        required=True,
        choices=list(ALLOWED),
        help="Which model/checkpoint to evaluate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model_type)
