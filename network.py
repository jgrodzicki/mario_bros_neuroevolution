import numpy as np
import operator
from functools import reduce
import torch
from torch import nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Network(nn.Module):
    def __init__(self, layers: nn.Module) -> None:
        super(Network, self).__init__()
        self.layers = layers

    def forward(self, state: torch.Tensor) -> int:
        output = self.layers.forward(state.view(1, 3, 240, 256))
        return max(min(int(output), 6), 0)

    def set_weights(self, weights: torch.Tensor) -> None:
        cpt = 0
        for param in self.parameters():
            tmp = reduce(operator.mul, param.size())

            param.data.copy_(weights[cpt:cpt + tmp].view(param.size()).to(device))
            cpt += tmp

    def get_weights(self) -> torch.Tensor:
        return torch.hstack([v.flatten() for v in self.parameters()])


DEFAULT_NETWORK = Network(layers=nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3)),
    nn.MaxPool2d(kernel_size=(3, 3)),
    nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3)),
    nn.MaxPool2d(kernel_size=(3, 3)),
    nn.Conv2d(in_channels=5, out_channels=1, kernel_size=(3, 3)),
    nn.MaxPool2d(kernel_size=(3, 3)),
    nn.Flatten(1, -1),
    nn.Linear(56, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
))
