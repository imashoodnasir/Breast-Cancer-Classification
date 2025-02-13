import numpy as np
import random

class ModifiedGreyWolfOptimizer:
    def __init__(self, obj_function, dim, population_size=10, max_iter=100, lb=-100, ub=100):
        self.obj_function = obj_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.a = 2  # Control parameter decreasing linearly

    def initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def adaptive_dynamic_parameter(self, i):
        """ Adaptive dynamic parameter control """
        return 2 * np.exp(-3 * (i / self.max_iter))

    def chaos_based_position_update(self, x, alpha, beta, delta):
        """ Introduces randomness for better global search """
        chaos_factor = np.random.rand(self.dim)
        return (alpha - self.a * chaos_factor * (alpha - x) +
                beta - self.a * chaos_factor * (beta - x) +
                delta - self.a * chaos_factor * (delta - x)) / 3

    def opposition_based_learning(self, x):
        """ Generates opposite solutions for better initialization """
        return self.lb + self.ub - x

    def optimize(self):
        population = self.initialize_population()
        fitness = np.array([self.obj_function(ind) for ind in population])
        sorted_indices = np.argsort(fitness)

        alpha, beta, delta = population[sorted_indices[:3]]  # Best three solutions

        for i in range(self.max_iter):
            self.a = self.adaptive_dynamic_parameter(i)

            for j in range(self.population_size):
                if np.random.rand() < 0.5:
                    population[j] = self.chaos_based_position_update(population[j], alpha, beta, delta)
                else:
                    opposition_solution = self.opposition_based_learning(population[j])
                    if self.obj_function(opposition_solution) < fitness[j]:
                        population[j] = opposition_solution

                population[j] = np.clip(population[j], self.lb, self.ub)

            fitness = np.array([self.obj_function(ind) for ind in population])
            sorted_indices = np.argsort(fitness)
            alpha, beta, delta = population[sorted_indices[:3]]  # Update top three

        return alpha, self.obj_function(alpha)  # Best solution and its fitness

# Example Feature Selection Objective Function
def feature_selection_objective(features):
    return np.sum(features ** 2)  # Example: Minimize squared sum

# Run mGWO for Feature Selection
optimizer = ModifiedGreyWolfOptimizer(obj_function=feature_selection_objective, dim=30, max_iter=100)
best_solution, best_fitness = optimizer.optimize()

print("Best Features:", best_solution)
print("Best Fitness Score:", best_fitness)
