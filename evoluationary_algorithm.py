from abc import abstractmethod
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import List
from tqdm import tqdm  # type: ignore
from nes_py.nes_env import NESEnv  # type: ignore

from individual import Individual
from network import Network


class EA:

    def __init__(
        self,
        env: NESEnv,
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
        self.network.set_weights(individual)

        for _ in range(self.eval_iters):
            action = self.network.forward(state)
            state, reward, done, info = self.env.step(action)

        return info['score']

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
        population_evals: List[float],
        children: List[Individual],
        children_evals: List[float]
    ) -> List[Individual]:
        raise NotImplementedError()
    
    @abstractmethod
    def parent_selection(self, population: List[Individual]) -> List[Individual]:
        raise NotImplementedError()
    
    def run(self) -> None:
        best_eval = -1e5
        
        for it in tqdm(range(self.max_iters), desc='Training', position=0, leave=True):
            population = self.random_population()
            evals = self.evaluate(individuals=population)
            
            parents = self.parent_selection(population=population)
            children = self.mutate(parents[:, :])
            children_evals = self.evaluate(individuals=children)
            
            population, evals = self.selection(
                population=population,
                population_evals=evals,
                children=children,
                children_evals=children_evals,
            )
            
            self.history_df.loc[len(self.history_df)] = [it, np.min(evals), np.mean(evals), np.min(evals)]
            
            if np.max(evals) > best_eval:
                best_eval = np.max(evals)
                np.save(f'models/{type(self).__name__}/{it}.npy', population[np.argmax(evals)], allow_pickle=True)
