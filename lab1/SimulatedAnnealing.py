import numpy as np
import random
import sys

class SimulatedAnnealing:
    def __init__(self, func, n, temperature=10000.0, cooling_rate=0.99, 
                 max_iter=50000, equilibrium=10, minimize=True):
        self.func = func
        self.n = n
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.equilibrium = equilibrium
        self.minimize = minimize
        
    def evaluate(self, solution):
        f = self.func(solution)
        if self.minimize:
            return f
        return -f
    
    def get_neighbour(self, solution):
        neighbour = solution + np.random.normal(0, 0.1, size=solution.shape)
        return neighbour

    def algorithm(self):
        solution = np.random.rand(self.n)
        best_solution = solution
        best_cost = self.evaluate(best_solution)
        for i in range(self.max_iter):
            for j in range(self.equilibrium):
                neighbour = self.get_neighbour(solution)
                energy = self.evaluate(neighbour) - self.evaluate(solution)
                if energy < 0:
                    solution = neighbour
                    best_solution = solution
                    best_cost = self.evaluate(solution)
                else:
                    p = np.exp(-energy/self.temperature)
                    if random.random() < p:
                        solution = neighbour
            self.temperature *= self.cooling_rate
            if self.temperature == 0:
                break
            if i % 1000 == 0:
                print(f'Iteration {i}: error = {best_cost}')
        return best_solution, best_cost
    
    
class SystemRegression():
    def __init__(self, file_path, max_iter):
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
        sa = SimulatedAnnealing(self.loss_function, 6, max_iter=self.max_iter)
        best_solution, best_cost = sa.algorithm()
        return best_solution, best_cost
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Neispravno pozvan program")
    else:
        max_iter = int(sys.argv[1])
        file_path = sys.argv[2]
        regression = SystemRegression(file_path, max_iter)
        best_solution, best_cost = regression.optimize()
        print(f'Best solution = {best_solution}, error = {best_cost}')