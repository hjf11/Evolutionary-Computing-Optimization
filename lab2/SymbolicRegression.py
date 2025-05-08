from GeneticProgramming import *
from tqdm import tqdm

class SymbolicRegression(GeneticProgramming):
    def __init__(self, file_path, operators = ['+', '-', '*', '/', 'sin', 'cos', 'sqrt', 'log', 'exp'],
                 max_depth=7, max_nodes=200, max_build_depth=6, population_size=500, tournament_size=7,
                 constant_range=None, max_gen = 100, max_loss_count=1000000, mutation_rate=0.14):
        
        super().__init__(file_path, operators, max_depth, max_nodes, max_build_depth, 
                 population_size, tournament_size, constant_range)
        
        self.max_gen = max_gen
        self.max_loss_count = max_loss_count
        
        self.reproduction_rate = 0.01
        self.mutation_rate = mutation_rate
        self.crossover_rate = 0.85
        
        self.best_found = None
    
    def algorithm(self):
        for i in tqdm(range(self.max_gen)):
            if self.loss_count >= self.max_loss_count:
                return
            new_generation = []
            sorted_population = sort_population(self.population)
            best_individual = sorted_population[0]
            if self.best_found is None or best_individual.fitness < self.best_found.fitness:
                tqdm.write(tree_to_string(best_individual.tree))
                tqdm.write(f'Loss: {best_individual.fitness}\n')
                self.best_found = best_individual
                if best_individual.fitness < 1e-15:
                    return
            new_generation.append(best_individual)    #elitizam: uvijek zadrzi najbolju
            while len(new_generation) < self.population_size:
                p = random.random()
                
                if p < self.reproduction_rate:
                    new_generation.append(self.selection(1))    #odaberi jednu jedinku koja ce se samo kopirati u novu generaciju
                    
                elif p < self.reproduction_rate + self.mutation_rate:
                    individual = self.selection(1)
                    mutated_individual = self.mutation(individual)
                    while mutated_individual == -1:
                        mutated_individual = self.mutation(individual)
                    new_generation.append(mutated_individual)
                    
                else:
                    parent1, parent2 = self.selection(2)
                    child = self.crossover(parent1, parent2)
                    while child == -1:
                        child = self.crossover(parent1, parent2)
                    new_generation.append(child)
                    
            self.population = new_generation
            
if __name__ == "__main__":
    sr = SymbolicRegression('lab2/f3.txt')
    sr.algorithm()
    print(f'Rješenje: {tree_to_string(sr.best_found)}\nLoss: {sr.best_found.fitness}')