import numpy as np
import random
import sys        

class MOOPProblem():   
    def get_number_of_objectives(self):
        raise NotImplementedError('This method should be overridden by subclasses.')
    
    def evaluate_solution(self, solution):
        raise NotImplementedError('This method should be overridden by subclasses.')

class Problem1(MOOPProblem):
    def get_number_of_objectives(self):
        return 4

    def evaluate_solution(self, solution):
        return [x ** 2 for x in solution]
    
class Problem2(MOOPProblem):
    def get_number_of_objectives(self):
        return 2

    def evaluate_solution(self, solution):
        x1, x2 = solution
        return [x1, (1 + x2) / x1]
    
class Individual:
    def __init__(self, solution):
        self.solution = solution
        self.objectives = None
        self.rank = None
        self.crowding_distance = 0.0
        self.domination_count = 0
        self.dominated_individuals = []
        
    def dominates(self, other):
        better_in_all = True
        strictly_better_in_one = False

        for z1, z2 in zip(self.objectives, other.objectives):
            if z1 < z2:
                strictly_better_in_one = True
            elif z1 > z2:
                better_in_all = False
        
        return better_in_all and strictly_better_in_one
        
    def __lt__(self, other):
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance
    
    def __str__(self):
        return f'{self.solution}, {self.objectives}'
    
    
class NSGA_II():
    def __init__(self, problem, bounds, population_size, max_iter, k=3):
        self.problem = problem
        self.bounds = bounds
        self.population_size = population_size
        self.k = k
        self.max_iter = max_iter
        
        self.population = []
        
    def initialize_population(self):
        population = []
        solutions = np.random.uniform(
            low=[b[0] for b in self.bounds], 
            high=[b[1] for b in self.bounds], 
            size=(self.population_size, len(self.bounds))
        )
        for solution in solutions:
            population.append(Individual(solution))
        return population
    
    def evaluate_population(self):
        for individual in self.population:
            individual.objectives = self.problem.evaluate_solution(individual.solution)
        
    def crowded_tournament_selection(self):
        tournament = random.sample(self.population, self.k)
        sorted_tournament = sorted(tournament)
        return sorted_tournament[0], sorted_tournament[1]
    
    # diskretno krizanje
    def crossover(self, parent1, parent2, eta=20):
        child_solution = [
            random.choice([p1, p2])
            for p1, p2 in zip(parent1.solution, parent2.solution)
        ]
        return Individual(np.clip(child_solution, [b[0] for b in self.bounds], [b[1] for b in self.bounds]))
    
    # uniformna mutacija
    def mutation(self, individual, mutation_rate=0.1):
        for i in range(len(individual.solution)):
            if random.random() < mutation_rate:
                individual.solution[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
        
    def create_new_population(self):
        population =[]
        while len(population) < self.population_size:
            parent1, parent2 = self.crowded_tournament_selection()
            child = self.crossover(parent1, parent2)
            self.mutation(child)        
            population.append(child)
        return population
    
    def crowding_distance(self, front):
        num_objectives = self.problem.get_number_of_objectives()

        for individual in front:
            individual.crowding_distance = 0.0

        for m in range(num_objectives):
            front.sort(key=lambda ind: ind.objectives[m])
            front[0].crowding_distance = front[-1].crowding_distance = float('inf')     # postavi rubne točke na inf
            f_min = front[0].objectives[m]
            f_max = front[-1].objectives[m]

            for i in range(1, len(front) - 1):
                if f_max != f_min:
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / (f_max - f_min)
        
    def non_dominated_sort(self):
        fronts = []

        front = []      # fronta 0
        for p in self.population:
            p.domination_count = 0
            p.dominated_individuals = []
            
            for q in self.population:
                if p != q:
                    if p.dominates(q):
                        p.dominated_individuals.append(q)
                    elif q.dominates(p):
                        p.domination_count += 1
                
            if p.domination_count == 0:
                p.rank = 0
                front.append(p)
        
        fronts.append(front)
        
        k = 0
        while len(fronts[k]) > 0:
            next_front = []
            for p in fronts[k]:
                for q in p.dominated_individuals:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = k + 1
                        next_front.append(q)
            fronts.append(next_front)
            k += 1

        fronts.pop()
        return fronts
            
    def algorithm(self):
        self.population = self.initialize_population()
        self.evaluate_population()

        for i in range(self.max_iter):
            new_population = self.create_new_population()
            self.population.extend(new_population)
            self.evaluate_population()

            fronts = self.non_dominated_sort()
            
            next_population = []
            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend(front)
                else:
                    self.crowding_distance(front)
                    front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                    next_population.extend(front[:self.population_size - len(next_population)])
                    break

            self.population = next_population
        
        pareto_front = fronts[0]
        pareto_objectives = [ind.objectives for ind in pareto_front]
        return pareto_front, pareto_objectives
            
            
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Neispravno pozvan program")
    else:
        problem = int(sys.argv[1])
        n = int(sys.argv[2])
        max_iter = int(sys.argv[3])
        
        if problem == 1:
            nsga_ii = NSGA_II(problem=Problem1(), bounds=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)], population_size=n, max_iter=max_iter)
        else:
            nsga_ii = NSGA_II(problem=Problem2(), bounds=[(0.1, 1), (0, 5)], population_size=n, max_iter=max_iter)
        
        pareto_front, pareto_objectives = nsga_ii.algorithm()
        # for ind, obj in zip(pareto_front, pareto_objectives):
        #     print(f"Solution: {ind.solution}, Objectives: {obj}")
            
        fronts = nsga_ii.non_dominated_sort()
        for i, front in enumerate(fronts):
            print(f"Front {i}: {len(front)} solutions")

        with open("izlaz-dec.txt", "w") as dec_file:
            for ind in nsga_ii.population:
                dec_file.write(" ".join(map(str, ind.solution)) + "\n")
                
        with open("izlaz-obj.txt", "w") as obj_file:
            for ind in nsga_ii.population:
                if problem == 2:
                    obj_file.write(f"({ind.objectives[0]}, {ind.objectives[1]})\n")
                else:
                    obj_file.write(" ".join(map(str, ind.objectives)) + "\n")

        if problem == 2:
            import matplotlib.pyplot as plt

            f1 = [obj[0] for obj in pareto_objectives]
            f2 = [obj[1] for obj in pareto_objectives]

            plt.figure(figsize=(8, 6))
            plt.scatter(f1, f2, c='blue', marker='o')
            plt.title("Pareto fronta problem 2")
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.xlim(0.1, 1)
            plt.ylim(0, 10)
            plt.grid()
            plt.savefig("pareto_front_problem2.png")
            plt.show()