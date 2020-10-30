from abc import abstractmethod
import numpy as np
import pandas as pd
from statistics import mean
from typing import List, Tuple
from tqdm import tqdm

from nes_py.wrappers import JoypadSpace

from individual import Individual
from network import Network


class EvolutionaryAlgorithm:

    def __init__(
        self,
        env: JoypadSpace,
        network: Network,
        population_size: int,
        individual_len: int,
        eval_iters: int,
        max_iters: int,
    ) -> None:
        self.env = env
        self.population_size = population_size
        self.individual_len = individual_len
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
        state = self.env.reset()
        self.network.set_weights(individual.weights)

        individual_reward: int = 0

        for _ in range(self.eval_iters):
            action = self.network.forward(state)
            state, reward, done, info = self.env.step(action)
            individual_reward += reward
            if done:
                self.env.reset()
                break

        return individual_reward

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
            children = self.mutate(parents)
            children_evals = self.evaluate(individuals=children)

            population, evals = self.selection(
                population=population,
                population_evals=evals,
                children=children,
                children_evals=children_evals,
            )

            self.history_df.loc[len(self.history_df)] = [it, min(evals), mean(evals), max(evals)]
            
            if max(evals) > best_eval:
                best_eval = max(evals)
                pbar.desc = f'Training (best eval: {best_eval} in {it} iter)'

                np.save(
                    f'models/{type(self).__name__}/best_inds/{it}.npy',
                    population[np.argmax(evals)],
                    allow_pickle=True,
                )

            if it % 50 == 0:
                self.history_df.to_csv(f'models/{type(self).__name__}/history_df.csv', index=False)

        pbar.close()
