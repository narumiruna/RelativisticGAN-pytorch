import glob
import os

from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.folder import pil_loader


class CatDataset(data.Dataset):

    def __init__(self, root: str, transform=None):
        self.root = root
        self.paths = glob.glob(os.path.join(self.root, '*/*aligned.jpg'))
        self.transform = transform

    def __getitem__(self, index: int):
        image = pil_loader(self.paths[index])

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.paths)


def get_cat_dataloader(root='data/cat', batch_size=64, num_workers=1, size=64):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor()])

    dataloader = data.DataLoader(
        CatDataset(root, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return dataloader
