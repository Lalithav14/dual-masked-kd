import torch

def generate_random_mask(img, mask_ratio=0.6):
    """
    Randomly masks input image patches
    """
    B, C, H, W = img.shape
    mask = torch.rand((B, 1, H, W), device=img.device)
    mask = (mask > mask_ratio).float()
    return mask
