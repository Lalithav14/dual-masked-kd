from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar100_loader(batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
