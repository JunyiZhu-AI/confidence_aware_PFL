import abc
import json
import os
import torch
import random
import numpy as np

__all__ = [
    "RandomDataSplit",
    "set_random",
    "save_args",
]


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class DataSplitStrategy(abc.ABC):

    def __init__(self, targets, n_classes, n_clients, n_labels, seed):
        self.targets = targets.cpu().numpy()
        self.n_classes = n_classes
        self.n_clients = n_clients
        self.n_labels = n_labels
        self.rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def assign(self, client_labels):
        """assign data to clients
        """


class RandomDataSplit(DataSplitStrategy):

    def assign(self, client_labels):
        """Randomly split the data and then distribute to clients.
        Args:
            client_labels: Possessed labels of each client
        """
        n_data = len(self.targets)
        cls_indices = []
        for i in range(self.n_classes):
            ind = np.arange(n_data)[self.targets == i]
            self.rng.shuffle(ind)
            cls_indices.append(ind)

        n_data_per_class = len(cls_indices[0])
        clipper = self.rng.choice(
            np.arange(1, n_data_per_class),
            self.n_clients * self.n_labels // self.n_classes - 1,
            replace=False
        )

        clipper.sort()
        cls_indices_block = [np.split(cid, clipper) for cid in cls_indices]

        client_indices = []
        for client in range(self.n_clients):
            indices = []
            for lbl in client_labels[client]:
                indices.append(cls_indices_block[lbl].pop())
            client_indices.append(np.concatenate(indices))

        assert np.all([len(x) == 0 for x in cls_indices_block])
        return client_indices


def save_args(**kwargs):
    if os.path.exists(os.path.join(kwargs["path_to_res"], "args.json")):
        os.remove(os.path.join(kwargs["path_to_res"], "args.json"))

    with open(os.path.join(kwargs["path_to_res"], "args.json"), "w") as f:
        json.dump(kwargs, f, indent=4)
