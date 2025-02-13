import numpy as np

class WhaleOptimizationAlgorithm:
    def __init__(self, cost_function, dim=10, population_size=20, max_iter=100):
        self.cost_function = cost_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.population = np.random.uniform(-10, 10, (population_size, dim))

    def optimize(self):
        best_solution = self.population[np.argmin([self.cost_function(ind) for ind in self.population])]

        for _ in range(self.max_iter):
            a = 2 - _ * (2 / self.max_iter)  # Linearly decreases a from 2 to 0

            for i in range(self.population_size):
                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * np.random.rand()
                b = 1
                l = (np.random.rand() - 0.5) * 2

                if np.random.rand() < 0.5:
                    if abs(A) < 1:
                        self.population[i] = best_solution - A * abs(C * best_solution - self.population[i])
                    else:
                        random_whale = self.population[np.random.randint(self.population_size)]
                        self.population[i] = random_whale - A * abs(C * random_whale - self.population[i])
                else:
                    self.population[i] = best_solution + b * l * abs(best_solution - self.population[i])

            best_solution = self.population[np.argmin([self.cost_function(ind) for ind in self.population])]

        return best_solution, self.cost_function(best_solution)

# Example Objective Function
def objective_function(solution):
    return np.sum(solution ** 2)

# Run WOA
woa = WhaleOptimizationAlgorithm(objective_function)
best_solution, best_cost = woa.optimize()
print("WOA Best Solution:", best_solution)
print("WOA Best Cost:", best_cost)
