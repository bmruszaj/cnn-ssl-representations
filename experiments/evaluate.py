import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any, Literal

from src.utils.params_yaml import load_yaml
from src.data.loaders import get_loaders

# ---------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
params: Dict[str, Any] = load_yaml()

ALLOWED: tuple[Literal[
    "baseline", "ae", "vae", "vae_pretrained", "simclr", "byol",
    "ae_frozen", "ae_unfrozen", "vae_frozen", "vae_unfrozen"
], ...] = (
    "baseline", "ae", "vae", "vae_pretrained", "simclr", "byol",
    "ae_frozen", "ae_unfrozen", "vae_frozen", "vae_unfrozen"
)


def evaluate(model_type: str) -> None:
    if model_type not in ALLOWED:
        raise ValueError(f"model_type must be one of {ALLOWED}")

    paths = params["paths"]
    ckpt_path = PROJECT_ROOT / paths["checkpoints_dir"] / f"{model_type}_model.pt"
    results_path = PROJECT_ROOT / paths["results_dir"] / f"{model_type}_acc.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders()

    if model_type == "baseline":
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, params["model"]["num_classes"])
        # load saved baseline weights
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
    else:
        # Load full model object saved during training (allow pickled models)
        model = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = model.to(device)
    model.eval()

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
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
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
