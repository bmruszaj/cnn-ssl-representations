import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from src.data.loaders import get_loaders
from src.utils.params_yaml import load_yaml
from src.models.resnet18_with_pools import build_resnet18
from src.models.vae import VAELoss

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
params: Dict[str, Any] = load_yaml()

def pretrain_vae():

    paths = params["paths"]

    save_path = PROJECT_ROOT / paths["checkpoints_dir"] / "vae_pretrained.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, _ = get_loaders()

    model = build_resnet18(
        pretrained=params['model']['pretrained'],
        pool_type='vae',
        freeze_pool=False,
        latent=params['model']['latent_channels'],
        num_classes=params['model']['num_classes'],
    ).to(device)

    vae = model.avgpool
    # Freeze all parameters except the VAE (avgpool)
    for name, p in model.named_parameters():
        if not name.startswith('avgpool'):
            p.requires_grad = False

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=params['vae']['train']['weight_decay'],
        weight_decay=0
    )
    criterion = VAELoss()

    epochs = params['vae']['pretrain']['epochs']

    for epoch in range(1, epochs + 1):

        vae.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'[VAE Pretrain] Epoch {epoch}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)

            # Extract late features (after layer4) without tracking gradients
            with torch.no_grad():
                x = model.conv1(images)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)

                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)

            feats = x  # 512-channel features from the last layer

            optimizer.zero_grad()
            recon = vae(feats)
            loss = criterion(vae, recon, feats)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'vae_loss': running_loss / (pbar.n + 1)})

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), save_path)

    print(f'Pretrained VAE saved to {save_path}')


if __name__ == '__main__':
    pretrain_vae()