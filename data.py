import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Callable, Optional


class CIFAR10Sub4(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        targets = np.array(self.targets)
        idx = np.argwhere(targets < 4)
        idx = np.random.choice(idx.reshape(-1), size=int(0.16*len(self)))
        self.targets = targets[idx]
        self.data = self.data[idx]


class Transforms:
    class CIFAR10:
        class VGG:

            train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        class ResNet:

            train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                    ),
                ]
            )

            test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                    ),
                ]
            )

    CIFAR100 = CIFAR10
    CIFAR10Sub4 = CIFAR10


def loaders(
    dataset,
    path,
    batch_size,
    num_workers,
    transform_name,
    use_test=False,
    shuffle_train=True,
):
    if dataset == "CIFAR10Sub4":
        ds = CIFAR10Sub4
    else:
        ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)

    if use_test:
        print("You are going to run models on the test set. Are you sure?")
        test_set = ds(path, train=False, download=True, transform=transform.test)
    else:
        print("Using train (45000) + validation (5000)")
        train_set.train_data = train_set.data[:-5000]
        train_set.train_labels = train_set.targets[:-5000]

        test_set = ds(path, train=True, download=True, transform=transform.test)
        test_set.train = False
        test_set.test_data = test_set.data[-5000:]
        test_set.test_labels = test_set.targets[-5000:]
        delattr(test_set, "data")
        delattr(test_set, "targets")
    print(f"Datasize: {len(train_set)}")
    print(f"Datasize: {len(test_set)}")
    return {
        "train": torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }, max(train_set.targets) + 1
