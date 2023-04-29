import copy
import torch
from model import *
from vem import VEMServer as Server
from vem.vem_client import VEMClient as Client
from vem import BinomialSampler
from utils import *


def train(
    batch_size,
    lr_base,
    lr_head,
    momentum,
    n_rounds,
    n_clients,
    n_labels,
    sampling_rate,
    dataset,
    model,
    n_mc,
    path_to_data,
    relabel,
    seed,
    max_data,
    head_epochs,
    base_epochs,
    scale,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random(seed)

    # prepare the dataset
    train_loaders, test_loaders, local_classes, client_labels, \
    class_2_superclass, client_test_ind, relabel = load_dataloaders(
        dataset=dataset,
        n_clients=n_clients,
        n_labels=n_labels,
        relabel=relabel,
        device=device,
        batch_size=batch_size,
        path_to_data=path_to_data,
        max_data=max_data,
        seed=seed
    )

    # build up the model
    model_base = eval(f"{model}Base()").to(device)
    client_sampler = BinomialSampler(n_clients, p=sampling_rate)
    server = Server(model_base=model_base,
                    device=device)
    server.mu_W = torch.randn(model_base.d_feature, local_classes, device=device)
    server.mu_b = torch.randn(local_classes, device=device)
    clients = [Client(
        device=device,
        model_base=copy.deepcopy(model_base),
        weight_mu=copy.deepcopy(server.mu_W),
        bias_mu=copy.deepcopy(server.mu_b),
        train_loader=train_loaders[n],
        client_labels=client_labels[n],
        momentum=momentum,
        scale=scale,
        n_mc=n_mc,
        class_2_superclass=class_2_superclass,
        base_epochs=base_epochs,
        head_epochs=head_epochs,
    ).to(device)
               for n in range(n_clients)]
    server.connect_clients(clients)

    for round_idx in range(n_rounds):
        train_losses = []
        for client_idx in range(n_clients):
            # personalization
            client = clients[client_idx]
            loss = client.train(mu_W=server.mu_W,
                                mu_b=server.mu_b,
                                lr_head=lr_head,
                                lr_base=lr_base,
                                relabel=relabel)
            train_losses.append(loss * len(client.train_loader.dataset) / server.num_data)

        train_losses_avg = sum(train_losses)
        print(f"Round {round_idx+1}, train loss = {train_losses_avg:.4f}")

        sampled_clients = client_sampler.sample()
        if round_idx in list(range(0, 100, 5)) + [99]:
            test_losses = []
            test_accs = []
            # test personalized models
            for client_idx in range(n_clients):
                client = clients[client_idx]
                test_loss, test_acc = client.test(test_loaders=[
                    test_loaders[idx] for idx in client_test_ind[client_idx]
                ], relabel=relabel)

                test_losses.append(test_loss)
                test_accs.append(test_acc)

            test_losses_avg = sum(test_losses) / len(test_losses)
            test_accs_avg = sum(test_accs) / len(test_accs)

            # update and evaluate the global model
            server.update_base_model(sampled_clients)
            server.update_center_model(sampled_clients)
            test_global_loss, test_global_acc = server.test(test_loaders=test_loaders,
                                                            relabel=relabel,
                                                            class_2_superclass=class_2_superclass)

            print(
                f"Round {round_idx+1}, " +
                f"PMs test loss = {test_losses_avg:.4f}, " +
                f"PMs test acc = {test_accs_avg:.4f}, " +
                f"GM test loss = {test_global_loss:.4f}, " +
                f"GM test acc = {test_global_acc:.4f}, "
            )

        else:
            # update the global model
            server.update_base_model(sampled_clients)
            server.update_center_model(sampled_clients)
