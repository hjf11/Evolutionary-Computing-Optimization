from SymbolicRegression import *
import sys

def load_config_from_file(file_path):
    with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("FunctionNodes:"):
                    operators = line.split(":")[1].strip().split(", ")
                elif line.startswith("ConstantRange:"):
                    constant_range_str = line.split(":")[1].strip()
                    if constant_range_str != "N/A":
                        constant_range = tuple(map(float, constant_range_str.split(", ")))
                    else:
                        constant_range = None
                elif line.startswith("PopulationSize:"):
                    population_size = int(line.split(":")[1].strip())
                elif line.startswith("TournamentSize:"):
                    tournament_size = int(line.split(":")[1].strip())
                elif line.startswith("CostEvaluations:"):
                    max_loss_count = int(line.split(":")[1].strip())
                elif line.startswith("MutationProbability:"):
                    mutation_rate = float(line.split(":")[1].strip())
                elif line.startswith("MaxTreeDepth:"):
                    max_depth = int(line.split(":")[1].strip())
    return operators, constant_range, population_size, tournament_size, max_loss_count, mutation_rate, max_depth

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        file_path = args[0]
        sr = SymbolicRegression(file_path)
    elif len(args) == 2:
        config_file, data_file = args
        operators, constant_range, population_size, tournament_size, max_loss_count, mutation_rate, max_depth = load_config_from_file(config_file)
        sr = SymbolicRegression(file_path=data_file, operators=operators, constant_range=constant_range, population_size=population_size,
                                tournament_size=tournament_size, max_loss_count=max_loss_count, max_depth=max_depth, mutation_rate=mutation_rate)
        
    sr.algorithm()
    print(f'Rješenje: {tree_to_string(sr.best_found.tree)}\nLoss: {sr.best_found.fitness}')