import numpy as np
import random
import sys

class Individual:
    def __init__(self, value, fitness):
        self.value = value
        self.fitness = fitness

# DE/rand/1/bin
# DE/best/1/bin
class DifferentialEvolution:
    def __init__(self, func, n, best, bounds=(-10, 10), population_size=100, max_iter=10000, crossover_rate=0.9):
        self.func = func
        self.n = n
        self.best = best
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        
        self.population = []
        
    def bin_crossover(self, i1, i2):
        new_individual = [v1 if random.random() < self.crossover_rate else v2 for v1, v2 in zip(i1, i2)]
        return new_individual

    def algorithm(self):
        # inicijaliziraj populaciju
        for i in range(self.population_size):
            value = np.random.uniform(self.bounds[0], self.bounds[1], self.n)
            score = self.func(value)
            individual = Individual(value, score)
            self.population.append(individual)
         
        best_idx = min(range(self.population_size), key=lambda j: self.population[j].fitness)
        best_solution = self.population[best_idx]
        
        # faktor skaliranja
        F = 0.8
        for i in range(self.max_iter):
            new_population = []
            for j, target in enumerate(self.population):
                # izbaci trenutnu jedinsku iz populacije (zato da ne odaberemo nju)
                temp_pop = self.population[:j] + self.population[j+1:]
                if self.best == 1:
                    r0 = best_solution
                    filtered_temp_pop = [ind for ind in temp_pop if ind != r0]
                    r1, r2 = random.sample(filtered_temp_pop, 2)
                else:
                    r0, r1, r2 = random.sample(temp_pop, 3)
                
                mutant = r0.value + F * (np.array(r1.value) - np.array(r2.value))
                trial_value = self.bin_crossover(mutant, target.value)
                # trial_value = np.clip(trial_value, self.bounds[0], self.bounds[1])
                trial_score = self.func(trial_value)
                
                if trial_score <= target.fitness:
                    trial = Individual(trial_value, trial_score)
                    new_population.append(trial)
                    if trial_score < best_solution.fitness:
                        best_solution = trial
                else:
                    new_population.append(target)
                    
            self.population = new_population
            
            if i % 500 == 0:
                print(f'Iteration {i}: error = {best_solution.fitness}')
            
        return best_solution.value, best_solution.fitness
    
class SystemRegression():
    def __init__(self, file_path, best, max_iter=5000):
        self.data = self.load_data(file_path)
        self.max_iter = max_iter
        self.best = best

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
        de = DifferentialEvolution(self.loss_function, 6, max_iter=self.max_iter, best=self.best)
        best_solution, best_cost = de.algorithm()
        return best_solution, best_cost
    
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Neispravno pozvan program")
    else:
        file_path = sys.argv[1]
        best = sys.argv[2]      # 0 za random, 1 za best
        regression = SystemRegression(file_path, best=best)
        best_solution, best_cost = regression.optimize()
        print(f'Best solution = {best_solution}, error = {best_cost}')