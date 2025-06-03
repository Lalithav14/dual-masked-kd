import matplotlib.pyplot as plt
import torch

def save_masked_image(img, mask, path):
    masked = img * mask
    grid = torch.cat([img, masked], dim=-1)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(path)

def cosine_similarity_loss(a, b):
    return 1 - torch.nn.functional.cosine_similarity(a, b).mean()
