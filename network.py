import numpy as np  # type: ignore
import torch
from torch import nn

from individual import Individual

CUDA = torch.cuda.is_available()


def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, requires_grad=False):
    x = torch.from_numpy(x)
    if CUDA:
        x = x.cuda()

    if requires_grad:
        return x.clone().contiguous().detach().requires_grad_(True)
    else:
        return x.clone().contiguous().detach()


class Network(nn.Module):
    def __init__(self, layers: nn.Module) -> None:
        super(Network, self).__init__()
        self.layers = layers

    def forward(self, state: np.ndarray) -> bytearray:
        return self.layers.forward(state)

    def set_weights(self, weights: Individual) -> None:
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(to_tensor(weights[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(to_tensor(weights[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_weights(self) -> Individual:
        return np.hstack([to_np(v).flatten() for v in self.parameters()])
