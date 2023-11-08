import pandas as pd
import os
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader


__all__ = [
    "load_sun397"
]


def load_sun397(n_clients,
                path,
                device,
                batch_size=20,
                seed=42):
    """
    prepare SUN397 data for personalized federated learning

    Args:
        n_clients: Number of clients.
        path: Path to dataset
        device: cuda or cpu
        batch_size: Mini-batchsize of training examples.
        seed: Random seed.
    """
    hierarchy = pd.read_excel(io=os.path.join(path, "SUN397", "hierarchy_three_levels", "three_levels.xlsx"),
                              sheet_name="SUN397",
                              header=1,
                              engine='openpyxl')
    super_classes = hierarchy["indoor"].to_numpy() + \
                    hierarchy["outdoor, natural"].to_numpy() + \
                    hierarchy["outdoor, man-made"].to_numpy()
    clean_classes = hierarchy["category"].to_numpy()[
        np.where(super_classes == 1)
    ]

    superlabels = np.zeros_like(clean_classes)
    indoor = hierarchy["indoor"].loc[hierarchy["category"].isin(clean_classes)].to_numpy().astype(bool)
    superlabels[indoor] = 0
    outdoor_natural = hierarchy["outdoor, natural"].loc[hierarchy["category"].isin(clean_classes)].to_numpy().astype(bool)
    superlabels[outdoor_natural] = 1
    outdoor_manmade = hierarchy["outdoor, man-made"].loc[hierarchy["category"].isin(clean_classes)].to_numpy().astype(bool)
    superlabels[outdoor_manmade] = 2
    short_clean_classes = np.array(
        [os.path.join(*Path(c).parts[2:])
         for c in clean_classes]
    )
    class_2_superclass = dict(zip(short_clean_classes, superlabels))
    superclass_2_class = dict()
    superclass_2_class[0] = short_clean_classes[indoor]
    superclass_2_class[1] = short_clean_classes[outdoor_natural]
    superclass_2_class[2] = short_clean_classes[outdoor_manmade]

    dataset = torchvision.datasets.SUN397(root=path,
                                          transform=transforms.ToTensor(),
                                          target_transform=torch.tensor)
    dataset._labels = np.asarray(dataset._labels)

    def class_idx(class_name):
        return np.where(
            dataset._labels == dataset.class_to_idx[class_name]
        )[0].tolist()

    idx_per_class = dict(
        zip(
            short_clean_classes,
            [class_idx(c) for c in short_clean_classes]
        )
    )

    clients_data_inds = []
    clients_labels = []
    test_datasets = {}
    rng = np.random.default_rng(seed)
    for client_idx in range(n_clients):
        num_data = round(rng.uniform(1, 67))
        labels = []
        data_inds = []
        for s in np.arange(3):
            num_data_per_class = np.asarray([len(idx_per_class[subc])
                                             for subc in superclass_2_class[s]])
            available_subclasses = superclass_2_class[s][num_data_per_class >= num_data]
            if len(available_subclasses) == 0:
                raise IndexError("All data in this super class has been consumed, "
                                 "please consider distribute less data to each client "
                                 "or create less clients")
            subc = rng.choice(available_subclasses)
            subc_idx = dataset.class_to_idx[subc]
            if subc_idx not in test_datasets:
                # testsize = min(round(len(idx_per_class[subc]) * 0.2), 100)
                test_datasets[subc_idx] = idx_per_class[subc][:30]
                idx_per_class[subc] = idx_per_class[subc][30:]

            labels.append(subc_idx)
            data_inds += idx_per_class[subc][:num_data]
            idx_per_class[subc] = idx_per_class[subc][num_data:]

        clients_data_inds.append(data_inds)
        clients_labels.append(labels)

    trainloaders, testloaders = load_data(dataset=dataset,
                                          test_datasets=test_datasets,
                                          clients_data_inds=clients_data_inds,
                                          batch_size=batch_size,
                                          device=device)

    return trainloaders, testloaders, clients_labels


def load_data(dataset,
              clients_data_inds,
              test_datasets,
              batch_size,
              device):
    train_loaders = []
    for client in clients_data_inds:
        data, targets = list(zip(
            *[dataset[i] for i in client]
        ))
        data = torch.stack(data).to(device)
        targets = torch.stack(targets).to(device)

        train_loaders.append(
            DataLoader(
                TensorDataset(data, targets),
                batch_size=batch_size,
                shuffle=True
            )
        )

    test_loaders = {}
    for c in test_datasets.keys():
        data, targets = list(zip(
            *[dataset[i] for i in test_datasets[c]]
        ))

        data = torch.stack(data).to(device)
        targets = torch.stack(targets).to(device)

        test_loaders[c] = DataLoader(TensorDataset(data, targets),
                                     batch_size=500,
                                     shuffle=False)

    return train_loaders, test_loaders


if __name__ == "__main__":
    load_sun397(n_clients=200,
                path="/home/junyi/data",
                device="cuda",
                batch_size=20,
                seed=42)
