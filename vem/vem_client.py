import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np


def make_prior(mu, rho):
    scale = rho * torch.ones_like(mu, device=mu.device)
    prior = D.Normal(mu, scale)
    return prior


class VEMClient(nn.Module):
    def __init__(
            self,
            device,
            model_base,
            weight_mu,
            bias_mu,
            train_loader,
            client_labels,
            momentum=0.9,
            scale=1,
            class_2_superclass=None,
            random_init=False,
            n_mc=5,
            base_epochs=10,
            head_epochs=10,):
        super(VEMClient, self).__init__()
        self.device = device
        self.client_labels = client_labels
        self.train_loader = train_loader
        self.class_2_superclass = class_2_superclass
        scale = torch.log(torch.expm1(torch.tensor(scale, dtype=torch.float))) if scale < 20 else scale
        if not random_init:
            self.weight_mu = nn.Parameter(weight_mu)
            self.weight_scale = nn.Parameter(torch.ones(weight_mu.shape).mul(scale))
            self.bias_mu = nn.Parameter(bias_mu)
            self.bias_scale = nn.Parameter(torch.ones(bias_mu.shape).mul(scale))
        else:
            self.weight_mu = nn.Parameter(torch.randn(weight_mu.shape))
            self.weight_scale = nn.Parameter(torch.randn(weight_mu.shape).mul(scale))
            self.bias_scale = nn.Parameter(torch.randn(bias_mu.shape).mul(scale))
        self.model_base = model_base
        self.var_params = [self.weight_mu,
                           self.bias_mu,
                           self.weight_scale,
                           self.bias_scale]
        self.head_epochs = head_epochs
        self.base_epochs = base_epochs
        self.rho = None
        self.momentum = momentum
        self.train_use_mc = True
        self.test_use_mc = False
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.n_mc = n_mc

    def test(self, test_loaders, relabel):
        if isinstance(test_loaders, list) or isinstance(test_loaders, np.ndarray):
            test_loaders = test_loaders
        else:
            test_loaders = [test_loaders]

        self.model_base.eval()
        test_loss = 0
        correct = 0
        W_post = D.Normal(self.weight_mu, F.softplus(self.weight_scale))
        b_post = D.Normal(self.bias_mu, F.softplus(self.bias_scale))
        preds_list = []
        targets_list = []
        num = 0
        with torch.no_grad():
            for test_loader in test_loaders:
                for data, target in test_loader:
                    if relabel:
                        target = torch.tensor([self.class_2_superclass[t.item()] for t in target],
                                              dtype=torch.int64)
                    inds = [t.cpu().item() in self.client_labels for t in target]
                    data, target = data[inds].to(self.device), target[inds].to(self.device)
                    z = self.model_base(data)
                    # Option 1: use mean of the variational approximation
                    if not self.test_use_mc:
                        output = torch.matmul(z, self.weight_mu).add(self.bias_mu)
                        test_loss += self.criterion(input=output, target=target).sum().item()
                        output = torch.softmax(output, dim=1)
                    # Option 2: use distribution of the variational approximation
                    else:
                        output = 0
                        for k in range(self.n_mc):
                            output += torch.softmax(
                                torch.matmul(z, W_post.sample()).add(b_post.sample()), dim=1)
                        output /= self.n_mc
                        output = torch.log(output)
                        test_loss += F.nll_loss(output, target, reduction='sum').item()

                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    num += target.numel()
                    preds_list += output.cpu().tolist()
                    targets_list += target.squeeze().cpu().tolist()

        test_loss /= num
        acc = correct / num
        return test_loss, acc

    def train(self, mu_W, mu_b, lr_head, lr_base, relabel):

        self.model_base.train()
        for p in self.var_params:
            p.requires_grad = True
        for p in self.model_base.parameters():
            p.requires_grad = False
        opt_local = torch.optim.SGD(self.var_params,
                                    lr=lr_head,
                                    momentum=self.momentum)
        var_params_copy = copy.deepcopy(self.var_params)

        # M-step: update rho
        self.rho = (F.softplus(self.weight_scale).square().sum() +
                    (self.weight_mu - mu_W).square().sum() +
                    F.softplus(self.bias_scale).square().sum() +
                    (self.bias_mu - mu_b).square().sum()
                    ).div(self.weight_mu.numel() + self.bias_mu.numel()).sqrt().item()
        W_prior = make_prior(mu_W, self.rho)
        b_prior = make_prior(mu_b, self.rho)
        train_loss = 0
        for epoch in range(self.head_epochs):
            data, target = zip(*[(d, t) for d, t in self.train_loader])
            data = torch.cat(data, dim=0)
            target = torch.cat(target, dim=0)
            if relabel:
                target = torch.tensor([self.class_2_superclass[t.item()] for t in target],
                                      dtype=torch.int64)

            opt_local.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                z = self.model_base(data)

            W_post = D.Normal(self.weight_mu, F.softplus(self.weight_scale))
            b_post = D.Normal(self.bias_mu, F.softplus(self.bias_scale))
            # use multiple mc samples to estimate the expecation
            nll = torch.tensor(0, dtype=torch.float).to(self.device)
            for k in range(self.n_mc):
                W = W_post.rsample()
                b = b_post.rsample()
                output = torch.matmul(z, W).add(b)
                loss = self.criterion(input=output, target=target)
                train_loss += loss.sum().div(self.n_mc).item()
                nll += loss.sum().div(self.n_mc)
            kl = D.kl_divergence(W_post, W_prior).sum() + D.kl_divergence(
                b_post, b_prior).sum()
            loss = nll + kl
            loss.backward()
            opt_local.step()
        train_loss /= (self.head_epochs * len(self.train_loader.dataset))

        # Update the base model
        opt_base = torch.optim.SGD(self.model_base.parameters(),
                                   lr=lr_base,
                                   momentum=self.momentum)
        for p in self.var_params:
            p.requires_grad = False
        for p in self.model_base.parameters():
            p.requires_grad = True
        W_post = D.Normal(self.weight_mu, F.softplus(self.weight_scale))
        b_post = D.Normal(self.bias_mu, F.softplus(self.bias_scale))
        for epoch in range(self.base_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if relabel:
                    target = torch.tensor([self.class_2_superclass[t.item()] for t in target],
                                          dtype=torch.int64)


                opt_base.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                z = self.model_base(data)
                # Option 1: use mean of the variational approximation
                if not self.train_use_mc:
                    output = torch.matmul(z, self.weight_mu).add(self.bias_mu)
                    loss = self.criterion(input=output, target=target.squeeze()).mean()
                # Option 2: use distribution of the variational approximation
                else:
                    loss = torch.tensor(0, dtype=torch.float).to(self.device)
                    for _ in range(self.n_mc):
                        output = torch.matmul(z, W_post.sample()).add(b_post.sample())
                        loss += self.criterion(input=output, target=target).mean().div(self.n_mc)

                loss.backward()
                opt_base.step()

        # Prevent invalid parameters from sending back to the server, although this barely happens.
        if np.isnan(train_loss):
            with torch.no_grad():
                for x, y in zip(self.var_params, var_params_copy):
                    x.data = y.data

        return train_loss
