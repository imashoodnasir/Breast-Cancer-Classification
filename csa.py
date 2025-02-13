import numpy as np

class CuckooSearchAlgorithm:
    def __init__(self, cost_function, dim=10, population_size=20, max_iter=100, pa=0.25):
        self.cost_function = cost_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.pa = pa
        self.population = np.random.uniform(-10, 10, (population_size, dim))

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        return np.random.randn(self.dim) * sigma

    def optimize(self):
        for _ in range(self.max_iter):
            new_population = self.population.copy()
            for i in range(self.population_size):
                step = self.levy_flight()
                new_population[i] += step * (self.population[np.random.randint(self.population_size)] - self.population[i])

                if self.cost_function(new_population[i]) < self.cost_function(self.population[i]):
                    self.population[i] = new_population[i]

                if np.random.rand() < self.pa:
                    self.population[i] = np.random.uniform(-10, 10, self.dim)

        best_solution = self.population[np.argmin([self.cost_function(ind) for ind in self.population])]
        return best_solution, self.cost_function(best_solution)

# Example Objective Function
def objective_function(solution):
    return np.sum(solution ** 2)

# Run CSA
csa = CuckooSearchAlgorithm(objective_function)
best_solution, best_cost = csa.optimize()
print("CSA Best Solution:", best_solution)
print("CSA Best Cost:", best_cost)
