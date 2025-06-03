import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from dual_masked_kd import MaskedAutoencoder, EMA

def random_mask(img, ratio=0.5):
    mask = torch.rand_like(img)
    mask = (mask > ratio).float()
    return mask

def train_step(model, teacher, optimizer, data):
    img, _ = data
    mask = random_mask(img)

    recon_s, feat_s = model(img, mask)
    recon_t, feat_t = teacher.forward(img, mask)

    loss_recon = F.mse_loss(recon_s, recon_t)
    loss_feat = F.cosine_embedding_loss(feat_s, feat_t, torch.ones(img.size(0)))

    loss = loss_recon + loss_feat
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    student = MaskedAutoencoder()
    teacher_model = MaskedAutoencoder()
    teacher = EMA(teacher_model)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    for epoch in range(2):
        for data in loader:
            loss = train_step(student, teacher, optimizer, data)
            print("Loss:", loss)
        teacher.update()

if __name__ == "__main__":
    main()
