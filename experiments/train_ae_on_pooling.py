import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from src.data.loaders import get_loaders
from src.utils.params_yaml import load_yaml
from src.models.resnet18_with_pools import build_resnet18
from src.models.ae import AELoss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
params: Dict[str, Any] = load_yaml()

def train_ae():
    paths = params["paths"]
    save_path = PROJECT_ROOT / paths["checkpoints_dir"] / "ae_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _ = get_loaders()

    model = build_resnet18(
        pretrained=params['model']['pretrained'],
        pool_type='ae',
        freeze_pool=params['ae']['freeze'],
        latent=params['model']['latent_channels'],
        num_classes=params['model']['num_classes'],
    ).to(device)

    # Freeze everything but avgpool (AE) and fc
    for name, p in model.named_parameters():
        if not (name.startswith('avgpool') or name.startswith('fc')):
            p.requires_grad = False

    optimizer = optim.AdamW(
        [
            {"params": model.avgpool.parameters()},
            {"params": model.fc.parameters()}
        ],
        lr=params['ae']['train']['learning_rate'],
        weight_decay=0
    )
    recon_criterion = AELoss(recon_loss_weight=params['ae']['recon_loss_weight'])
    cls_criterion   = nn.CrossEntropyLoss()
    cls_weight      = params['ae']['train'].get('cls_loss_weight', 1.0)

    epochs = params['ae']['train']['epochs']
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'[AE Train] Epoch {epoch}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # 1) extract early features
            feats = model.relu(model.bn1(model.conv1(images)))

            # 2) AE forward: get reconstructed features (and z, which we ignore here)
            recon_feats, _ = model.avgpool(feats)

            # 3) classification head—run recon through the rest of ResNet:
            x = recon_feats
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            logits = model.fc(x)

            # 4) losses
            loss_recon = recon_criterion(recon_feats, feats)
            loss_cls   = cls_criterion(logits, labels) * cls_weight
            loss = loss_recon + loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    torch.save(model.state_dict(), save_path)
    print(f'AE model saved to {save_path}')

if __name__ == '__main__':
    train_ae()
