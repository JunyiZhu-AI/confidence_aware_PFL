import torch.nn as nn

__all__ = [
    "CNNCifar100Base",
]


class CNNCifar100Base(nn.Module):
    def __init__(self):
        super(CNNCifar100Base, self).__init__()
        act = nn.ReLU()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(256, 100, kernel_size=3, stride=1, padding=1),
            act,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.d_feature = 100

    def forward(self, x):
        return self.body(x).squeeze()
