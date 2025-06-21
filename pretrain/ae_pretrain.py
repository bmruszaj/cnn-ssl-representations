import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from src.data.loaders import get_loaders
from src.utils.params_yaml import load_yaml
from src.models.resnet18_with_pools import build_resnet18
from src.models.ae import AELoss  # your AE loss from src/models/ae.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
params: Dict[str, Any] = load_yaml()

def pretrain_ae():
    save_path = PROJECT_ROOT / params["paths"]["checkpoints_dir"] / "ae_pretrained.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _ = get_loaders()

    model = build_resnet18(
        pretrained=params['model']['pretrained'],
        pool_type='ae',
        freeze_pool=False,
        latent=params['model']['latent_channels'],
        num_classes=params['model']['num_classes'],
    ).to(device)

    ae = model.avgpool
    for name, p in model.named_parameters():
        if not name.startswith('avgpool'):
            p.requires_grad = False

    optimizer = optim.AdamW(
        ae.parameters(),
        lr=params['ae']['train']['learning_rate'],
        weight_decay=0
    )
    criterion = AELoss(recon_loss_weight=params['ae']['recon_loss_weight'])

    epochs = params['ae']['train']['epochs']
    for epoch in range(1, epochs + 1):
        ae.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'[AE Pretrain] Epoch {epoch}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)

            with torch.no_grad():
                feats = model.relu(model.bn1(model.conv1(images)))

            optimizer.zero_grad()
            recon, _ = ae(feats)
            loss = criterion(recon, feats)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'ae_loss': running_loss / (pbar.n + 1)})

    torch.save(ae.state_dict(), save_path)
    print(f'Pretrained AE saved to {save_path}')

if __name__ == '__main__':
    pretrain_ae()
