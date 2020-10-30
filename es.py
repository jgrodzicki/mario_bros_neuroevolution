import numpy as np
from typing import List, Tuple

from nes_py.wrappers import JoypadSpace

from evoluationary_algorithm import EvolutionaryAlgorithm
from individual import Individual
from network import Network


class ES(EvolutionaryAlgorithm):

    def __init__(
        self,
        env: JoypadSpace,
        network: Network,
        population_size: int,
        individual_len: int,
        eval_iters: int,
        max_iters: int,
        mutation_chance: float,
    ):
        super().__init__(env, network, population_size, individual_len, eval_iters, max_iters)
        self.mutation_chance = mutation_chance

    def random_population(self) -> List[Individual]:
        population = [Individual(weights=(np.random.rand(self.individual_len) - 0.5) * 10)
                      for _ in range(self.population_size)]
        return population

    def mutate(self, parents: List[Individual]) -> List[Individual]:
        ind_idxs, gene_idxs = np.where(np.random.rand(len(parents), self.individual_len) <= self.mutation_chance)

        for ind_idx, gene_idx in np.stack((ind_idxs, gene_idxs), axis=1):
            parents[ind_idx].weights[gene_idx] += (np.random.rand()-0.5) * 3

        return parents

    def crossover(self, ind_1: Individual, ind_2: Individual) -> List[Individual]:
        new_weights_1 = ind_1.weights
        new_weights_2 = ind_2.weights

        selected_idxs = np.where(np.random.rand(self.individual_len) < 0.5)

        new_weights_1[selected_idxs] = ind_2.weights[selected_idxs]
        new_weights_2[selected_idxs] = ind_1.weights[selected_idxs]

        return [
            Individual(weights=new_weights_1),
            Individual(weights=new_weights_2),
        ]

    def selection(
        self,
        population: List[Individual],
        population_evals: List[int],
        children: List[Individual],
        children_evals: List[int],
    ) -> Tuple[List[Individual], List[int]]:
        evals = np.append(population_evals, children_evals)
        all_individuals = np.stack((population, children), axis=1)

        idxs = np.argsort(evals)[-self.population_size:]
        selected_individuals = all_individuals[idxs]
        selected_evals = evals[idxs]

        return selected_individuals, selected_evals

    def parent_selection(self, population: List[Individual], evals: List[int]) -> List[Individual]:
        idxs = np.argsort(evals)[-self.population_size // 2:]
        selected_individuals = population[idxs]

        return selected_individuals
