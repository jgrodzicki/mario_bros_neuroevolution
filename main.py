import torch

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from es import ES
from individual import Individual
from network import DEFAULT_NETWORK, device


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    network = DEFAULT_NETWORK.to(device)

    agent = ES(
        env=env,
        network=network,
        population_size=100,
        individual_len=997,
        eval_maps=3,
        eval_iters=1000,
        max_iters=50,
        mutation_chance=0.01,
    )

    agent.run()


if __name__ == '__main__':
    main()
