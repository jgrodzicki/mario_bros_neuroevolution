import numpy as np
import torch
from typing import NamedTuple

from nes_py.wrappers import JoypadSpace

from network import Network, DEFAULT_NETWORK

Individual = NamedTuple('Individual', (
    ('weights', np.ndarray),
))


def present_individual(
    env: JoypadSpace,
    individual: Individual,
) -> None:
    network: Network = DEFAULT_NETWORK
    network.set_weights(individual)

    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        action = network.forward(torch.Tensor(state.copy()).cuda())
        state, reward, done, info = env.step(action)
        env.render()

    env.close()
