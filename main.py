from torch import nn

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from es import ES
from network import Network, CUDA, DEFAULT_NETWORK


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    network = DEFAULT_NETWORK

    if CUDA:
        network = network.cuda()

    agent = ES(
        env=env,
        network=network,
        population_size=50,
        individual_len=997,
        eval_iters=500,
        max_iters=5,
        mutation_chance=0.001,
    )

    agent.run()


if __name__ == '__main__':
    main()
