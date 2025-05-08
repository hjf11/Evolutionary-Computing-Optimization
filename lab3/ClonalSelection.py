import numpy as np
import random
import sys

class Individual:
    def __init__(self, value, affinity):
        self.value = value
        self.affinity = affinity
        self.age = 0
        
    def __str__(self):
        return f'{self.value}, affinity: {self.affinity}'

class ClonalSelection:
    def __init__(self, func, n, population_size=50, bounds=(-10, 10), num_of_cloned_inds=10, beta=5, rho=0.2, max_survival_iter=1000, max_iter=10000):
        self.func = func
        self.n = n
        self.population_size = population_size
        self.bounds = bounds
        self.num_of_cloned_inds = num_of_cloned_inds
        self.d = 0.1 * population_size
        self.beta = beta
        self.rho = rho
        self.max_survival_iter = max_survival_iter
        self.max_iter = max_iter
        
        self.population = []
    
    # roulette wheel selection
    def selection(self):
        total_affinity = sum(1 / ind.affinity for ind in self.population)
        probabilities = [(1 / ind.affinity) / total_affinity for ind in self.population]

        selected = []
        for _ in range(self.num_of_cloned_inds):
            r = random.random()
            cumulative_probability = 0
            for individual, probability in zip(self.population, probabilities):
                cumulative_probability += probability
                if r <= cumulative_probability:
                    selected.append(individual)
                    break
        return selected
    
    def generate_clones(self, clones):
        clones.sort(key=lambda ind: ind.affinity)
        cloned_population = []
        for i, ind in enumerate(clones):
            num_clones = int((self.beta * self.num_of_cloned_inds) / (i+1))
            cloned_population.extend([Individual(ind.value.copy(), ind.affinity) for _ in range(num_clones)])
        return cloned_population
    
    # uniformna mutacija
    def mutation(self, individual, mutation_rate):
        new_value = np.array([v + np.random.uniform(-1, 1) if random.random() < mutation_rate else v for v in individual.value])
        new_affinity = self.func(new_value)
        individual.value = new_value
        individual.affinity = new_affinity
        return individual
    
    def hipermutation(self, cloned_population):
        for ind in cloned_population:
            # affinity je zapravo error pa je zato 1 - p
            mutation_rate = 1 - np.exp(-self.rho * ind.affinity)
            ind = self.mutation(ind, mutation_rate)
        return cloned_population
        
    def algorithm(self):
        # stvori inicijalnu populaciju
        for i in range(self.population_size):
            value = np.random.uniform(self.bounds[0], self.bounds[1], self.n)
            affinity = self.func(value)
            individual = Individual(value, affinity)
            self.population.append(individual)
            
        best_individual = min(self.population, key=lambda ind: ind.affinity)
            
        for i in range(self.max_iter):
            selected_for_cloning = self.selection()

            cloned_population = self.generate_clones(selected_for_cloning)
            cloned_population = self.hipermutation(cloned_population)
            
            cloned_population.sort(key=lambda ind: ind.affinity)
            new_population = cloned_population[:int(self.population_size-self.d)]
            
            # dodaj d random jedinki
            while len(new_population) < self.population_size:
                value = np.random.uniform(self.bounds[0], self.bounds[1], self.n)
                affinity = self.func(value)
                individual = Individual(value, affinity)
                new_population.append(individual)
            
            self.population = new_population
            new_best_individual = min(self.population, key=lambda ind: ind.affinity)
            if new_best_individual.affinity < best_individual.affinity:
                best_individual = new_best_individual
            if i % 500 == 0:
                print(f'Iteration {i}: error = {new_best_individual.affinity}')
        return best_individual.value, best_individual.affinity


class SystemRegression():
    def __init__(self, file_path, max_iter=10000):
        self.data = self.load_data(file_path)
        self.max_iter = max_iter

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue                
                
                values = line.strip().strip("[]").split(",")
                values = list(map(float, values))
                x_values = values[:-1]
                y_value = values[-1]
                data.append((np.array(x_values), y_value))
        return data
    
    def transfer_characteristic(self, x, params):
        a, b, c, d, e, f = params
        x1, x2, x3, x4, x5 = x
        return a*x1 + b*(x1**3)*x2 + c*np.exp(d*x3)*(1 + np.cos(e*x4)) + f*(x4*(x5**2))
    
    def loss_function(self, params):
        error = 0.0
        for x, y in self.data:
            y_pred = self.transfer_characteristic(x, params)
            error += (y - y_pred) ** 2
        return error / len(self.data)
    
    def optimize(self):
        cs = ClonalSelection(self.loss_function, 6, max_iter=self.max_iter)
        best_solution, best_cost = cs.algorithm()
        return best_solution, best_cost
    
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Neispravno pozvan program")
    else:
        file_path = sys.argv[1]
        regression = SystemRegression(file_path)
        best_solution, best_cost = regression.optimize()
        print(f'Best solution = {best_solution}, error = {best_cost}')