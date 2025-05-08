import math
import numpy as np

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
class Individual:        
    def __init__(self, node:Node, fitness = None):
        self.tree = node
        self.fitness = fitness 
    
    def update_fitness(self, loss_function, data):
        self.fitness = loss_function(self.tree, data)
    
    def __str__(self):
        print_tree(self.tree)
        return f'Fitness: {self.fitness}'
    
def evaluate_expression(node, row):
    expression = tree_to_string(node)
    variables = row.to_dict()
    safe_namespace = {
        'sin': math.sin,
        'cos': math.cos,
        'sqrt': lambda x: math.sqrt(x) if x >= 0 else 1,
        'log': lambda x: math.log10(x) if x > 0 else 1,
        'exp': math.exp
    }
    safe_namespace.update(variables)
    
    try:
        return eval(expression, {"__builtins__": None}, safe_namespace)
    except (ZeroDivisionError, ValueError, OverflowError):
        return 1

def loss_function(node, data):
    losses = []
    for _, row in data.iterrows():
        predicted = evaluate_expression(node, row)
        loss = (predicted - row.iloc[-1]) ** 2
        losses.append(loss)
    return np.mean(losses)

def sort_population(population):
    return sorted(population, key=lambda individual: individual.fitness)
    
def print_tree(node, level=0, prefix='root: '):  
    if node is not None:
        print(" " * (4 * level) + prefix + str(node.value))
        if node.left or node.right:
            print_tree(node.left, level+1, prefix="L--- ")
            print_tree(node.right, level+1, prefix="R--- ")
    
def count_nodes(node):
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)

def node_level(root, node):
    if not root:
        return None
    queue = [(root, 0)]
    while queue:
        current, level = queue.pop(0)
        if current == node:
            return level
        if current.left:
            queue.append((current.left, level + 1))
        if current.right:
            queue.append((current.right, level + 1))
    return None

def tree_depth(node):
    if not node:
        return 0
    left_depth = tree_depth(node.left)
    right_depth = tree_depth(node.right)
    return max(left_depth, right_depth) + 1
 
def copy_tree(node):
    if node is None:
        return None
    new_node = Node(node.value)
    new_node.left = copy_tree(node.left)
    new_node.right = copy_tree(node.right)
    return new_node

#vraca cvor, njegogovg roditelja i je li lijevo ili desno dijete
#lijevo dijete: 0; desno dijete: 1
def get_node_at_index(root, index):
    if not root:
        return None, None, None
    queue = [(root, None, None)]
    current_index = 0
    while queue:
        node, parent, child_type = queue.pop(0)
        if current_index == index:
            return node, parent, child_type
        if node.left:
            queue.append((node.left, node, 0))  # lijevo dijete
        if node.right:
            queue.append((node.right, node, 1))  # desno dijete
        current_index += 1
    return None, None, None  
    
def tree_to_string(node):
    if node is None:
        return ""
    if node.left is None and node.right is None:
        return str(node.value)
    if node.right is None:
        return f"{node.value}({tree_to_string(node.left)})"
    return f"({tree_to_string(node.left)} {node.value} {tree_to_string(node.right)})"