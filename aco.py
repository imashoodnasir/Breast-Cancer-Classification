import numpy as np

class AntColonyOptimization:
    def __init__(self, cost_function, num_ants=10, num_iterations=100, alpha=1, beta=2, evaporation_rate=0.5, q=10):
        self.cost_function = cost_function
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.num_nodes = 10  # Example number of decision variables
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))

    def optimize(self):
        best_solution = None
        best_cost = float('inf')

        for _ in range(self.num_iterations):
            solutions = []
            costs = []

            for _ in range(self.num_ants):
                solution = np.random.permutation(self.num_nodes)
                cost = self.cost_function(solution)
                solutions.append(solution)
                costs.append(cost)

                if cost < best_cost:
                    best_solution, best_cost = solution, cost

            self.pheromone *= (1 - self.evaporation_rate)

            for solution, cost in zip(solutions, costs):
                for i in range(len(solution) - 1):
                    self.pheromone[solution[i], solution[i+1]] += self.q / cost

        return best_solution, best_cost

# Example Objective Function
def objective_function(solution):
    return np.sum(solution ** 2)

# Run ACO
aco = AntColonyOptimization(objective_function)
best_solution, best_cost = aco.optimize()
print("ACO Best Solution:", best_solution)
print("ACO Best Cost:", best_cost)
