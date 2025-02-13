import numpy as np

class BatAlgorithm:
    def __init__(self, cost_function, dim=10, population_size=20, max_iter=100, freq_min=0, freq_max=2, alpha=0.9, gamma=0.9):
        self.cost_function = cost_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.alpha = alpha
        self.gamma = gamma
        self.population = np.random.uniform(-10, 10, (population_size, dim))
        self.velocities = np.zeros((population_size, dim))
        self.frequencies = np.zeros(population_size)
        self.loudness = np.ones(population_size)
        self.pulse_rate = np.random.uniform(0, 1, population_size)
        self.best_solution = self.population[np.argmin([self.cost_function(ind) for ind in self.population])]

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.population_size):
                self.frequencies[i] = self.freq_min + (self.freq_max - self.freq_min) * np.random.rand()
                self.velocities[i] += (self.population[i] - self.best_solution) * self.frequencies[i]
                new_solution = self.population[i] + self.velocities[i]

                if np.random.rand() > self.pulse_rate[i]:
                    new_solution = self.best_solution + 0.01 * np.random.randn(self.dim)

                new_cost = self.cost_function(new_solution)
                if new_cost < self.cost_function(self.population[i]) and np.random.rand() < self.loudness[i]:
                    self.population[i] = new_solution
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] = self.pulse_rate[i] * (1 - np.exp(-self.gamma * t))

                if new_cost < self.cost_function(self.best_solution):
                    self.best_solution = new_solution

        return self.best_solution, self.cost_function(self.best_solution)

# Example Objective Function
def objective_function(solution):
    return np.sum(solution ** 2)

# Run BA
ba = BatAlgorithm(objective_function)
best_solution, best_cost = ba.optimize()
print("BA Best Solution:", best_solution)
print("BA Best Cost:", best_cost)
