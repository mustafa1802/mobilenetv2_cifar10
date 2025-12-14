import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
):
    """
    Returns trainloader, testloader for CIFAR-10 with augmentations.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


def make_calib_loader(test_dataset, batch_size: int = 128, num_samples: int = 1024):
    """
    Build a smaller calibration loader from the test dataset.
    """
    indices = list(range(min(num_samples, len(test_dataset))))
    calib_ds = Subset(test_dataset, indices)
    calib_loader = torch.utils.data.DataLoader(
        calib_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return calib_loader
