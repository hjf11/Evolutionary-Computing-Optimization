import numpy as np
import random
import sys

class Particle:
    def __init__(self, position, velocity, score):
        self.position = position
        self.velocity = velocity
        self.best_local_position = position
        self.best_local_score = score
        self.best_global_position = None
        self.best_global_score = None
        
    def __str__(self):
        return f'Me: {self.position}, {self.velocity}, best local: {self.best_local_position}, {self.best_local_score}'


class ParticleSwarmOptimization:
    def __init__(self, func, n, position_bounds=(-10, 10), velocity_bounds=(-1, 1), swarm_size=30, max_iter=50000, k=None):
        self.func = func
        self.n = n
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.k = k
        
        self.particles = []
        
    def evaluate(self, solution):
        return self.func(solution)
    
    def update_neighbourhood(self):
        k = int(self.k)
        for i, p in enumerate(self.particles):
            neighbourhood_indices = [(i + offset) % self.swarm_size for offset in range(-k, k+1)]
            
            best_particle = min((self.particles[j] for j in neighbourhood_indices), key=lambda particle: particle.best_local_score)
            
            p.best_global_position = best_particle.best_local_position
            p.best_global_score = best_particle.best_local_score
        
    def algorithm(self):
        # inizijaliziraj sve čestice
        for i in range(self.swarm_size):
            pos = np.random.uniform(self.position_bounds[0], self.position_bounds[1], self.n)
            v = np.random.uniform(self.velocity_bounds[0], self.velocity_bounds[1], self.n)
            score = self.func(pos)
            p = Particle(position=pos, velocity=v, score=score)
            self.particles.append(p)
        
        if self.k is not None:
            self.update_neighbourhood()

        # Nađi za sada najbolje globalno rješenje
        best_global_particle_idx = min(range(self.swarm_size), key=lambda j: self.particles[j].best_local_score)
        best_global_position = self.particles[best_global_particle_idx].best_local_position
        best_global_score = self.particles[best_global_particle_idx].best_local_score

        # hiperparametri
        w = (i / self.max_iter) * (0.4 - 0.8) + 0.8  # Inercija
        c1 = 2  # Kognitivna stopa učenja
        c2 = 2  # Socijalna stopa učenja

        for i in range(self.max_iter):
            for p in self.particles:
                if self.k is None:      # globalno susjedstvo
                    p.velocity = (w * p.velocity + c1 * random.random() * (p.best_local_position - p.position) 
                                + c2 * random.random() * (best_global_position - p.position))
                else:
                    p.velocity = (w * p.velocity + c1 * random.random() * (p.best_local_position - p.position) 
                                + c2 * random.random() * (p.best_global_position - p.position))
                p.velocity = np.clip(p.velocity, self.velocity_bounds[0], self.velocity_bounds[1])
                
                p.position += p.velocity
                p.position = np.clip(p.position, self.position_bounds[0], self.position_bounds[1])
            
                score = self.func(p.position)

                # Ažuriraj lokalno najbolje rješenje
                if score < p.best_local_score:
                    p.best_local_score = score
                    p.best_local_position = p.position
                
                if self.k is not None:
                    # Ažuriraj svoje globalno najbolje rješenje
                    if score < p.best_global_score:
                        p.best_global_position = p.position
                        p.best_global_score = score
                        self.update_neighbourhood()

                # Ažuriraj globalno najbolje rješenje
                if score < best_global_score:
                    best_global_position = p.position
                    best_global_score = score

            if i % 1000 == 0:
                print(f'Iteration {i}: error = {best_global_score}')

        return best_global_position, best_global_score

       
class SystemRegression():
    def __init__(self, file_path, max_iter=50000, k=None):
        self.data = self.load_data(file_path)
        self.max_iter = max_iter
        self.k = k

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
        pso = ParticleSwarmOptimization(self.loss_function, 6, max_iter=self.max_iter, k=self.k)
        best_solution, best_cost = pso.algorithm()
        return best_solution, best_cost
    
if __name__ == "__main__":
    k = None
    if len(sys.argv) not in (2, 3):
        print("Neispravno pozvan program")
        exit
    if len(sys.argv) == 3:
        k = sys.argv[2]

    file_path = sys.argv[1]
    regression = SystemRegression(file_path, k=k)
    best_solution, best_cost = regression.optimize()
    print(f'Best solution = {best_solution}, error = {best_cost}')