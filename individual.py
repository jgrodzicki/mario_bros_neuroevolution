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
    ) -> None:
        network.set_weights(self.weights)

        done = True
        for step in range(5000):
            if done:
                state = env.reset()
            action = network.forward(torch.Tensor(state.copy()).to(device))
            state, reward, done, info = env.step(action)
            env.render()

        env.close()
