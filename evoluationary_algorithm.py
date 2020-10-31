from abc import abstractmethod
import numpy as np
import pandas as pd
from statistics import mean
import torch
from typing import List, Tuple
from tqdm import tqdm

from nes_py.wrappers import JoypadSpace

from individual import Individual
from network import Network, device


class EvolutionaryAlgorithm:

    def __init__(
        self,
        env: JoypadSpace,
        network: Network,
        population_size: int,
        individual_len: int,
        eval_maps: int,
        eval_iters: int,
        max_iters: int,
    ) -> None:
        self.env = env
        self.population_size = population_size
        self.individual_len = individual_len
        self.eval_maps = eval_maps
        self.eval_iters = eval_iters
        self.max_iters = max_iters
        self.history_df = pd.DataFrame(columns=['iteration', 'min eval', 'mean eval', 'max eval'])
        self.network = network
    
    @abstractmethod
    def random_population(self) -> List[Individual]:
        raise NotImplementedError()

    def evaluate(self, individuals: List[Individual]) -> List[int]:
        return list(map(lambda ind: self._evaluate_individual(individual=ind), individuals))

    def _evaluate_individual(self, individual: Individual) -> int:
        self.network.set_weights(individual.weights)

        cumulative_rewards = [0] * self.eval_maps

        for map_i in range(self.eval_maps):
            current_cumulative_reward = 0
            state = self.env.reset()

            for _ in range(self.eval_iters):
                action = self.network.forward(torch.Tensor(state.copy()).to(device))
                state, reward, done, info = self.env.step(action)
                current_cumulative_reward += reward
                if done:
                    break

            cumulative_rewards[map_i] = current_cumulative_reward

        return int(np.cbrt(np.prod(cumulative_rewards)))

    @abstractmethod
    def mutate(self, parents: List[Individual]) -> List[Individual]:
        raise NotImplementedError()

    @abstractmethod
    def crossover(self, ind_1: Individual, ind_2: Individual) -> List[Individual]:
        raise NotImplementedError()

    @abstractmethod
    def selection(
        self,
        population: List[Individual],
        population_evals: List[int],
        children: List[Individual],
        children_evals: List[int]
    ) -> Tuple[List[Individual], List[int]]:
        raise NotImplementedError()

    @abstractmethod
    def parent_selection(self, population: List[Individual], evals: List[int]) -> List[Individual]:
        raise NotImplementedError()
    
    def run(self) -> None:
        best_eval = -1e5

        population = self.random_population()
        evals = self.evaluate(individuals=population)

        pbar = tqdm(total=self.max_iters, desc='Training', position=0, leave=True)

        for it in range(self.max_iters):
            pbar.update(1)

            parents = self.parent_selection(population=population, evals=evals)
            children = [Individual(weights=torch.Tensor([])) for _ in range(len(parents))]
            for i in range(0, len(parents), 2):
                children[i], children[i + 1] = self.crossover(parents[i], parents[i + 1])

            children = self.mutate(children)
            children_evals = self.evaluate(individuals=children)

            population, evals = self.selection(
                population=population,
                population_evals=evals,
                children=children,
                children_evals=children_evals,
            )

            self.history_df.loc[len(self.history_df)] = [it, min(evals), mean(evals), max(evals)]

            if max(evals) > best_eval:
                best_at_it = it
                best_eval = max(evals)

                torch.save(
                    population[np.argmax(evals)].weights,
                    f'models/{type(self).__name__}/best_inds/{it}.tch',
                )

            if (it + 1) % 50 == 0:
                self.history_df.to_csv(f'models/{type(self).__name__}/history_df.csv', index=False)

            pbar.desc = f'Training (evals: {min(evals)} | {mean(evals)} | {max(evals)}, best at: {best_at_it} it)'

        pbar.close()
