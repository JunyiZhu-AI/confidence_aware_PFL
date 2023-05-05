import torch
import numpy as np
import torch.nn as nn
import copy


class VEMServer:

    def __init__(self, model_base, beta, device):
        self.accum_model_state = None
        self.mu_W = None
        self.mu_b = None
        self.clients = []
        self.num_data = 0
        self.model_base = model_base
        self.device = device
        self.beta = beta

    def connect_clients(self, clients):
        self.clients = clients
        self.num_data = np.sum(len(x.train_loader.dataset) for x in self.clients)

    @torch.no_grad()
    def receive(self, model_base_state, coef):
        if self.accum_model_state is None:
            self.accum_model_state = copy.deepcopy(model_base_state)
            for key in self.accum_model_state.keys():
                self.accum_model_state[key].mul_(coef)

        else:
            for key in self.accum_model_state.keys():
                self.accum_model_state[key].add_(
                    model_base_state[key].mul(coef)
                )

    @torch.no_grad()
    def aggregation(self):
        model_base = self.model_base.state_dict()
        for key in self.accum_model_state.keys():
            self.accum_model_state[key].mul_(1 - self.beta).add_(
                model_base[key].mul(self.beta)
            )

    @torch.no_grad()
    def update_center_model(self, sampled_clients):
        weight_sum = 0
        w_sum = torch.zeros_like(self.mu_W)
        b_sum = torch.zeros_like(self.mu_b)

        for client_idx in sampled_clients:
            client = self.clients[client_idx]
            weight_mu, bias_mu, _, _ = client.var_params
            w_sum.add_(weight_mu.div(client.rho**2))
            b_sum.add_(bias_mu.div(client.rho**2))
            weight_sum += 1/client.rho**2

        self.mu_W = self.beta * self.mu_W + (1 - self.beta) * w_sum.div(weight_sum)
        self.mu_b = self.beta * self.mu_b + (1 - self.beta) * b_sum.div(weight_sum)

    @torch.no_grad()
    def update_base_model(self, sampled_clients):
        sampled_data = np.sum(
            len(self.clients[x].train_loader.dataset)
            for x in sampled_clients
        )
        for client_idx in sampled_clients:
            self.receive(
                self.clients[client_idx].model_base.state_dict(),
                len(self.clients[client_idx].train_loader.dataset)/sampled_data
            )

        self.aggregation()
        for client in self.clients:
            client.model_base.load_state_dict(self.accum_model_state)
        self.model_base.load_state_dict(self.accum_model_state)
        self.accum_model_state = None

    def test(self, test_loaders, class_2_superclass, relabel):
        if isinstance(test_loaders, list) or isinstance(test_loaders, np.ndarray):
            test_loaders = test_loaders
        else:
            test_loaders = [test_loaders]

        self.model_base.eval()
        test_loss = 0
        correct = 0
        preds_list = []
        targets_list = []
        criterion = nn.CrossEntropyLoss(reduction="none")
        with torch.no_grad():
            for test_loader in test_loaders:
                for data, target in test_loader:
                    if relabel:
                        target = torch.tensor([class_2_superclass[t.item()] for t in target],
                                              dtype=torch.int64)

                    data, target = data.to(self.device), target.to(self.device)
                    z = self.model_base(data)
                    output = torch.matmul(z, self.mu_W).add(self.mu_b)
                    test_loss += criterion(input=output, target=target).sum().item()
                    output = torch.softmax(output, dim=1)
                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()
                    preds_list += output.cpu().tolist()
                    targets_list += target.squeeze().cpu().tolist()

        test_loss /= np.sum(len(ld.dataset) for ld in test_loaders)
        acc = correct / np.sum(len(ld.dataset) for ld in test_loaders)
        return test_loss, acc
