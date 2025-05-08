from Utils import *
import random
import pandas as pd

class GeneticProgramming:
    def __init__(self, file_path, operators, max_depth=20, max_nodes=200, max_build_depth=6, 
                 population_size=500, tournament_size=7, constant_range = None):
        self.data = self.load_data(file_path)
        self.terminals = self.data.columns[:-1].tolist()
        self.constant_range = constant_range
        
        self.operators = operators
        self.operator_arity = {'+': 2,
                               '-': 2,
                               '*': 2,
                               '/': 2,
                               'sin': 1,
                               'cos': 1,
                               'sqrt': 1,
                               'log': 1,
                               'exp': 1}
        
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_build_depth = max_build_depth
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.loss_count = 0
        
        self.population = self.ramped_half_and_half()
        self.update_fitnesses()
        
    def load_data(self, file_path):
        data = pd.read_csv(file_path, sep='\t', header=None)
        data.columns = [f'x{i}' for i in range(1, data.shape[1])] + ['y']
        return data

    def create_leaf(self):
        if self.constant_range is not None:
            if random.random() < 0.5:
                low, high = self.constant_range
                # constant = random.randint(low, high)
                constant = random.uniform(low, high)
                return Node(constant)
        return Node(random.choice(self.terminals))
    
    def full(self, depth, level=0):
        if depth == 0:
            return self.create_leaf()
        operator = random.choice(self.operators)
        node = Node(operator)
        arity = self.operator_arity[operator]
        if arity >= 1:
            node.left = self.full(depth - 1, level + 1)
        if arity == 2:
            node.right = self.full(depth - 1, level + 1)
        return node

    def grow(self, depth, level=0):
        if depth == 0 or (level != 0 and random.random() < 0.5):
            return self.create_leaf()
        operator = random.choice(self.operators)
        node = Node(operator)
        arity = self.operator_arity[operator]
        if arity >= 1:
            node.left = self.grow(depth - 1, level + 1)
        if arity == 2:
            node.right = self.grow(depth - 1, level + 1)
        return node

    def ramped_half_and_half(self):
        population = []
        depths = list(range(2, self.max_build_depth-1))
        individuals_num = self.population_size // (2*len(depths))
        for depth in depths:
            for i in range(individuals_num):
                node = self.full(depth)
                population.append(Individual(node))
                node = self.grow(depth)
                population.append(Individual(node))
                
        #ako velicina populacije nije lijepo dijeljiva, ostatak popuni nasumicno
        while len(population) < self.population_size:
            if random.random() < 0.5:
                population.append(Individual(self.full(random.choice(depths))))
            else:
                population.append(Individual(self.grow(random.choice(depths))))
        return population
        
    def loss_function(self, node):
        self.loss_count += 1
        return loss_function(node, self.data)
    
    def update_fitnesses(self):
        for individual in self.population:
            self.loss_count += 1
            individual.update_fitness(loss_function, self.data)
    
    #7-turnirska
    def selection(self, num_of_individuals):
        tournament = random.sample(self.population,self. tournament_size)
        sorted_tournament = sort_population(tournament)
        if num_of_individuals == 1:
            return sorted_tournament[0]
        else:
            return sorted_tournament[0], sorted_tournament[1]

    def check_validity(self, tree):
        if count_nodes(tree) > self.max_nodes:
            return False
        if tree_depth(tree) > self.max_depth:
            return False
        return True
    
    #zamjena podstabla
    def crossover(self, parent1, parent2):
        child1 = copy_tree(parent1.tree)
        child2 = copy_tree(parent2.tree)
        
        idx1 = random.randint(1, count_nodes(parent1.tree)-1)  #uniformno bira nasumican index, od 1 jer ne želimo uzeti korijen
        idx2 = random.randint(1, count_nodes(parent2.tree)-1)
        
        node1, p1, right1 = get_node_at_index(child1, idx1)
        node2, p2, right2 = get_node_at_index(child2, idx2)
        
        if not right1:
            p1.left = node2
        elif right1:
            p1.right = node2
        if not right2:
            p2.left = node1
        elif right2:
            p2.right = node1
            
        fitness1 = self.loss_function(child1)
        fitness2 = self.loss_function(child2)
        if self.check_validity(child1) and self.check_validity(child2):
            if fitness1 <= fitness2:
                return Individual(child1, fitness1)
            return Individual(child2, fitness2) 
        if self.check_validity(child1):
            return Individual(child1, fitness1)
        if self.check_validity(child2):
            return Individual(child2, fitness2)
        return -1 
            
    #zamjena cvora podstablom
    def mutation(self, individual):
        tree = copy_tree(individual.tree)
        idx = random.randint(1, count_nodes(tree)-1)
        n, p, right = get_node_at_index(tree, idx)
        depth = node_level(tree, n)   #izracunaj na kojoj je dubini odabrani cvor
        max_d = self.max_depth - depth
        d = random.randint(0, max_d)    #odaberi random dubinu novog podstabla
        new_node = self.grow(d)
        if not right:
            p.left = new_node
        else:
            p.right = new_node
        if self.check_validity(tree):
            fitness = self.loss_function(tree)
            return Individual(tree, fitness)
        return -1