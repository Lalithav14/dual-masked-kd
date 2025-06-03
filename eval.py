from dual_masked_kd import MaskedAutoencoder, EMA
from dataset import get_cifar100_loader
from masking import generate_random_mask
import torch

def evaluate(model, ema_teacher):
    loader = get_cifar100_loader(batch_size=8)
    model.eval()
    ema_teacher.model.eval()

    total_loss = 0
    with torch.no_grad():
        for img, _ in loader:
            mask = generate_random_mask(img)
            _, feat_s = model(img, mask)
            _, feat_t = ema_teacher.forward(img, mask)

            loss = torch.nn.functional.mse_loss(feat_s, feat_t)
            total_loss += loss.item()
    print("Avg Feature MSE:", total_loss / len(loader))
