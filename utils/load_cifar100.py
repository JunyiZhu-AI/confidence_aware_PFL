import numpy as np
from .utils import *
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader


__all__ = [
    "load_cifar100",
    "superclass"
]


superclass = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
              ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
              ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
              ['bottle', 'bowl', 'can', 'cup', 'plate'],
              ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
              ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
              ['bed', 'chair', 'couch', 'table', 'wardrobe'],
              ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
              ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
              ['bridge', 'castle', 'house', 'road', 'skyscraper'],
              ['cloud', 'forest', 'mountain', 'plain', 'sea'],
              ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
              ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
              ['crab', 'lobster', 'snail', 'spider', 'worm'],
              ['baby', 'boy', 'girl', 'man', 'woman'],
              ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
              ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
              ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
              ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
              ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]


def load_cifar100(n_clients,
                  path,
                  device,
                  batch_size=20,
                  seed=42):
    if n_clients % 5 != 0:
        raise ValueError("# of clients must be exactly divided by 5.")

    def get_dataset(train):
        dataset = torchvision.datasets.CIFAR100(root=path,
                                                train=train,
                                                download=True,
                                                transform=cifar100_transform())
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        data, targets = next(iter(dataloader))
        return data, targets, dataset.class_to_idx

    train_data, train_targets, class_to_idx = get_dataset(True)
    test_data, test_targets, _ = get_dataset(False)

    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)

    return load_data(train_data=train_data,
                     train_targets=train_targets,
                     test_data=test_data,
                     test_targets=test_targets,
                     class_to_idx=class_to_idx,
                     n_classes=100,
                     n_clients=n_clients,
                     batch_size=batch_size,
                     seed=seed)


def load_data(train_data,
              train_targets,
              test_data,
              test_targets,
              n_classes,
              n_clients,
              batch_size,
              class_to_idx,
              seed=1234):
    client_train_datasets, client_labels, cls_2_supercls = get_split_datasets(data=train_data,
                                                                              targets=train_targets,
                                                                              n_clients=n_clients,
                                                                              n_classes=n_classes,
                                                                              class_to_idx=class_to_idx,
                                                                              seed=seed)

    test_datasets = []
    test_targets_np = test_targets.cpu().numpy()
    for i in range(n_classes):
        cls_ind = np.arange(len(test_data))[test_targets_np == i]
        test_datasets.append(
            TensorDataset(test_data[cls_ind], test_targets[cls_ind])
        )

    train_loaders = np.array([
        DataLoader(d, batch_size=batch_size, shuffle=True) for d in client_train_datasets
    ])
    test_loaders = np.array([
        DataLoader(d, batch_size=500, shuffle=False) for d in test_datasets
    ])

    return train_loaders, test_loaders, client_labels, cls_2_supercls


class ClientDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform):
        super(ClientDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def get_split_datasets(
        data,
        targets,
        n_clients,
        n_classes,
        class_to_idx,
        seed,
):
    # Initialize clients' targets
    rng = np.random.default_rng(seed)
    func = lambda x: [class_to_idx[y] for y in x]
    superclass_to_class = [func(x) for x in superclass]
    client_labels = []

    for _ in range(n_clients // 5):
        for c in superclass_to_class:
            rng.shuffle(c)
        client_labels += list(zip(*superclass_to_class))

    split_strategy = RandomDataSplit(targets=targets,
                                     n_classes=n_classes,
                                     n_clients=n_clients,
                                     n_labels=20,
                                     seed=seed)

    client_indices = split_strategy.assign(client_labels)

    client_datasets = []
    for client in range(n_clients):
        ind = client_indices[client]
        client_data = data[ind]
        client_targets = targets[ind]
        client_d = TensorDataset(client_data, client_targets)
        client_d.class_to_index = class_to_idx
        client_datasets.append(client_d)

    # class to superclass mapping
    cls_2_supercls = {}
    for i in range(len(superclass_to_class)):
        for c in superclass_to_class[i]:
            cls_2_supercls[c] = i

    return client_datasets, client_labels, cls_2_supercls


def cifar100_transform():
    return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5074, 0.4867, 0.4411),
                                     (0.2011, 0.1987, 0.2025)),
            ])
