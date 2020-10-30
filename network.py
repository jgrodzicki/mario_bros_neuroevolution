import numpy as np
from typing import Union
import torch
from torch import nn

from individual import Individual

CUDA = torch.cuda.is_available()


def to_np(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_tensor(x: Union[np.ndarray, torch.Tensor], requires_grad: bool = False) -> torch.Tensor:
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

    def forward(self, state: np.ndarray) -> int:
        output = self.layers.forward(torch.Tensor(state.copy()).view(1, 3, 240, 256))
        return max(min(int(output), 6), 0)

    def set_weights(self, weights: np.ndarray) -> None:
        cpt = 0
        for param in self.parameters():
            tmp = np.prod(param.size())

            if torch.cuda.is_available():
                param.data.copy_(to_tensor(weights[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(to_tensor(weights[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_weights(self) -> Individual:
        return Individual(weights=np.hstack([to_np(v).flatten() for v in self.parameters()]))
