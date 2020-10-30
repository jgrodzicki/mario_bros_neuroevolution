from torch import nn

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from es import ES
from network import Network


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    network = Network(layers=nn.Sequential(
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

    agent = ES(
        env=env,
        network=network,
        population_size=10,
        individual_len=997,
        eval_iters=500,
        max_iters=1,
        mutation_chance=0.001,
    )

    # agent.run()
    pop = agent.random_population()
    agent.evaluate(pop)


if __name__ == '__main__':
    main()
