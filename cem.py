import numpy as np
import pandas as pd
import pickle
import torch
from typing import List
from tqdm import trange
import os

from evoluationary_algorithm import EvolutionaryAlgorithm
from individual import Individual
from network import device


class CEM(EvolutionaryAlgorithm):
    
    def __init__(
        self,
        env,
        network,
        individual_len,
        max_iters,
        eval_maps,
        eval_iters,
        sigma_st,
        sigma_end,
        tau,
        population_size,
        elitist=None,
        batch_size=100,
    ):
        super().__init__(env, network, population_size, individual_len, eval_maps, eval_iters, max_iters)
        self.sigma_st = sigma_st
        self.sigma_end = sigma_end
        self.tau = tau
        self.elitist = elitist
        self.batch_size = batch_size
        
        if self.elitist is None:
            self.elitist = int(self.population_size // 2)
        
        self.params_df = pd.DataFrame({'iteration': [], 'mean sigma': [], 'max sigma': []})
    
    
    def _draw_population(self, mu: float, sigma: torch.Tensor) -> List[Individual]:
        # pop = torch.zeros((self.population_size, self.individual_len))
        pop = []  # type: List[Individual]
        for i in range(self.population_size):
            pop.append(Individual(weights=torch.normal(mu, torch.sqrt(sigma)).to(device)))
        return pop
    
    
    def run(self):
        mu = torch.ones(self.individual_len).to(device)
        sigma = self.sigma_st.to(device)
        epsilon = 2
        
        total_steps = actor_steps = 0
        
        best, best_eval, best_at_it = None, -np.inf, -1
        
        pbar = trange(self.max_iters, desc='Training', position=0, leave=True)
        
        start_it = 0
        if os.path.exists('saved_cem.pkl'):
            saved_cem = pickle.load(open('saved_cem.pkl', 'rb'))
            start_it = saved_cem['iteration']
            mu = saved_cem['params']['mu']
            sigma = saved_cem['params']['sigma']
            epsilon = saved_cem['params']['epsilon']
            
            best, best_eval = saved_cem['best'], saved_cem['best_eval']
            
            self.history_df = saved_cem['history_df']
            self.params_df = saved_cem['params_df']
            
            pbar.update(start_it)
        
        for it in range(start_it, self.max_iters):
            population = self._draw_population(mu, sigma)
            
            evals = self.evaluate(population)
            
            np_evals = np.array(evals)
            
            mu, sigma = self._update_params(population, evals, mu, sigma, epsilon)
            epsilon = self.tau * epsilon + (1 - self.tau) * (self.sigma_end.mean())
            
            if np_evals.max() > best_eval:
                best, best_eval, best_at_it = population[np.argmax(evals)], np_evals.max(), it
            
            # if it % 5 == 0:
            self.history_df.loc[it] = [it, np_evals.min(), np_evals.mean(), np_evals.max()]
            self.params_df.loc[it] = [it, sigma.mean().item(), sigma.max().item()]
            
            if total_steps % 5 == 0:
                to_write = {
                    'iteration': it,
                    'params': {'mu': mu, 'sigma': sigma, 'epsilon': epsilon},
                    'population': population,
                    'best': best,
                    'best_eval': best_eval,
                    'history_df': self.history_df,
                    'params_df': self.params_df,
                }
                pickle.dump(to_write, open(f'saved_cem.pkl', 'w+b'),
                            pickle.HIGHEST_PROTOCOL)
            
            #     best, best_eval = None, -np.inf
            
            pbar.desc = f'Training (evals: {np_evals.min()} | {np_evals.mean()} | {np_evals.max()}, best at: {best_at_it} it)'
            pbar.update(1)
        
        to_write = {
            'iteration': it,
            'params': {'mu': mu, 'sigma': sigma, 'epsilon': epsilon},
            'population': population,
            'best': best,
            'best_eval': best_eval,
            'history_df': self.history_df,
            'params_df': self.params_df,
        }
        pickle.dump(to_write, open(f'saved_cem.pkl', 'w+b'),
                    pickle.HIGHEST_PROTOCOL)
        
        return best
    
    
    def _update_params(self, pop, evals, old_mu, old_sigma, epsilon):
        lambdas = torch.Tensor(
            [torch.log(torch.Tensor([1 + self.elitist])) / (1 + i) for i in range(self.elitist)]).reshape(-1, 1).to(
            device)
        lambdas /= (lambdas.sum() + 1e-10)
        #         lambdas = np.full(self.elitist, 1/self.elitist)
        idxs = np.argsort(evals)[::-1][:self.elitist]
        z = []
        for idx in idxs:
            z.append(pop[idx].weights)
        z = torch.stack(z)
        
        # mu = torch.sum(torch.Tensor([list(lambdas[i]*z[i]) for i in range(self.elitist)]), dim=0)
        mu = torch.sum(lambdas * z, dim=0)
        # sigma = torch.sum(torch.Tensor([list(lambdas[i]*(z[i] - old_mu)**2) for i in range(self.elitist)]), dim=0)
        sigma = torch.sum(lambdas * (z - old_mu), dim=0)
        sigma += epsilon
        sigma = torch.clip(sigma, 0.4, 20)
        return mu, sigma