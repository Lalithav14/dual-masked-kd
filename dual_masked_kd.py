import torch
import torch.nn as nn
from torchvision import models

class MaskedAutoencoder(nn.Module):
    def __init__(self, backbone='vit_b_16'):
        super().__init__()
        self.encoder = models.__dict__[backbone](pretrained=True)
        self.decoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
        )
    
    def forward(self, x, mask):
        x = x * mask  # Apply masking
        features = self.encoder(x)
        recon = self.decoder(features)
        return recon, features

class EMA:
    def __init__(self, model, beta=0.99):
        self.model = model
        self.shadow = {name: param.clone() for name, param in model.named_parameters()}
        self.beta = beta

    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = self.beta * self.shadow[name] + (1 - self.beta) * param.data

    def forward(self, x, mask):
        with torch.no_grad():
            x = x * mask
            features = self.model.encoder(x)
            recon = self.model.decoder(features)
        return recon, features
