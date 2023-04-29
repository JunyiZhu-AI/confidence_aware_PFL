from utils.load_cifar10 import load_cifar10
from utils.load_cifar100 import load_cifar100
import copy


def load_dataloaders(
        dataset,
        n_clients,
        n_labels,
        relabel,
        device,
        batch_size,
        path_to_data,
        seed,
        max_data=None,
):
    if dataset == "CIFAR10":
        train_loaders, test_loaders, client_labels = load_cifar10(
            n_clients,
            n_labels=n_labels,
            batch_size=batch_size,
            path=path_to_data,
            seed=seed,
            max_data=max_data,
            device=device
        )
        client_test_ind = copy.deepcopy(client_labels)
        if relabel:
            local_classes = n_labels
        else:
            local_classes = 10
        class_2_superclass = None
    elif dataset == "CIFAR100":
        if n_labels != 20 or not relabel:
            Warning("CIFAR100 datasets must be relabeled to 20 superclasses!")
        local_classes = 20
        relabel = True

        train_loaders, test_loaders, client_labels, class_2_superclass = load_cifar100(
            n_clients=n_clients,
            path=path_to_data,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        client_test_ind = []
        for cl in client_labels:
            client_test_ind.append(list(cl))
    else:
        raise NotImplementedError

    return train_loaders, test_loaders, local_classes, client_labels, class_2_superclass, client_test_ind, relabel
