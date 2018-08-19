from torch.utils import data
from torchvision import datasets, transforms


def get_dataloader(root='data', batch_size=64, num_workers=0, size=64):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor()])

    dataloader = data.DataLoader(
        datasets.MNIST(root, train=True, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return dataloader
