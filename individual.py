import numpy as np
import time
import torch

from nes_py.wrappers import JoypadSpace

from network import Network, device


class Individual:

    def __init__(self, weights: torch.Tensor):
        self.weights = weights

    def present_individual(
        self,
        env: JoypadSpace,
        network: Network,
        iters: int = 5000
    ) -> None:
        network.set_weights(self.weights)

        done = True
        for step in range(iters):
            if done:
                state = env.reset()
                # for _ in range(10):
                #     state, reward, done, info = env.step(env.action_space.sample())
            action = network.forward(torch.Tensor(state.copy()).to(device))
            state, reward, done, info = env.step(action)
            env.render()
            # time.sleep(0.02)

        env.close()
