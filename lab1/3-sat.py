import sys
import numpy as np
import random
from tqdm import tqdm

def read_file(file_path):    
    try:
        with open(file_path, 'r') as file:
            formula = []
            for line in file:
                # ignoriraj komentare
                if line.startswith('c'):
                    continue
                
                elif line.startswith('p'):
                    parts = line.split()
                    if len(parts) >= 4:
                        var_num = int(parts[2])
                        clause_num = int(parts[3])
                    else:
                        raise ValueError("Neispravno definirani broj varijabli i broj klauzula")
                
                # ignoriraj sve ispod %
                elif line.startswith('%'):
                    break
                
                # definicije klauzula
                else:
                    parts = line.split()
                    clause = []
                    for part in parts:
                        num = int(part)

                        if num == 0:
                            formula.append(clause)
                            clause = []
                        else:
                            clause.append(num)
                    if clause:
                        formula.append(clause)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        
    return var_num, clause_num, formula

#za brute force algoritam, vraća idući binarni broj po redu
def next_binary(variables):
    bin_string = ''.join(map(str, variables.astype(int)))    #pretvori polje u string
    bin_num = int(bin_string, 2)
    bin_num += 1
    bin_string = bin(bin_num)[2:].zfill(len(variables))
    return np.array([bit == '1' for bit in bin_string], dtype=bool)

#vraca boolean polje velicine broja klauzula koje ima 1 ako je klauzula zadovoljena
def calculate_clauses(variables, formula):
    clauses = []
    for clause in formula:
        satisfied = False
        for var in clause:
            if var > 0:
                satisfied = satisfied or variables[np.abs(var)-1]
            else:
                satisfied = satisfied or not variables[np.abs(var)-1]   
        clauses.append(satisfied)
    return clauses

#algoritam 1
def brute_force(var_num, formula):
    final_res = None
    variables = np.zeros(var_num, dtype=bool)
    
    for i in range(2**var_num):
        clauses = calculate_clauses(variables, formula)
        res = clauses[0]
        for clause in clauses[1:]:
            res = res and clause
            
        if res:
            print(np.array(variables, dtype=int))
            final_res = variables
            
        #promijeni kombinaciju varijabli
        variables = next_binary(variables)
            
    return final_res

#prva fitness funkcija
def number_of_satisfied(solution):
    return sum(solution)

#funkcija koja vraća susjedstvo
def pi_function(x):
    neighbours = []
    for i in range(len(x)):
        new_x = x.copy()
        new_x[i] = not new_x[i]
        neighbours.append(new_x)
    return neighbours

#za printanje
def to_string(variables):
    return ''.join(map(str, (np.array(variables, dtype=int))))

#algoritam 2
def iterative_search(var_num, formula, max_iter=100000):
    variables = np.random.choice([True, False], size=var_num)
    for i in tqdm(range(max_iter)):
        clauses = calculate_clauses(variables, formula)
        fitness = number_of_satisfied(clauses)
        
        #provjeri da nismo već u globalnom optimumu
        if fitness == len(clauses):
            return variables
        
        neighbours = pi_function(variables)
        fitnesses = []
        for neighbour in neighbours:
            neighbour_clauses = calculate_clauses(neighbour, formula)
            fitnesses.append(number_of_satisfied(neighbour_clauses))
        max_fit = max(fitnesses)

        #jesmo li u lokalnom minimumu
        if max_fit < fitness:
            print(f'Neuspjeh! Lokalni minimum: {to_string(variables)}')
            return None
        #izvuci sve kandidate s maksimalnim fitnessom
        indices = [i for i, v in enumerate(fitnesses) if v == max_fit]
        candidates = [neighbours[i] for i in indices]
        #izaberi jedan na random
        index = random.randrange(len(candidates))
        variables = candidates[index]
        #provjeri je li to globalni optimum
        if max_fit == len(clauses):
            return variables
    print(f'Dosegnut max_iter, najbolje pronađen rezultat: {to_string(variables)}, dobrota: {fitness}/{len(clauses)}')
    return None

#prema zadanoj formuli azurira post
def update_post(clauses, post, percentage_up, percentage_down):
    for i, clause in enumerate(clauses):
        if clause:
            post[i] += (1-post[i])*percentage_up
        else:
            post[i] += (0-post[i])*percentage_down
    return post

#fitness za algoritam 3        
def fitness_function(solution, post, percentage_unit_amount):
    z = sum(solution)
    for i, clause in enumerate(solution):
        if clause:
            z += percentage_unit_amount * (1-post[i])
        else:
            z -= percentage_unit_amount * (1-post[i])
    return z   
    
#algoritam 3
def modified_iter_search(var_num, clause_num, formula, max_iter=100000, num_of_best=2, percentage_up=0.01, percentage_down=0.1, percentage_unit_amount=50):
    post = np.zeros(clause_num, dtype=float)
    variables = np.random.choice([True, False], size=var_num)
    for i in tqdm(range(max_iter)):
        clauses = calculate_clauses(variables, formula)
        fitness = number_of_satisfied(clauses)
        #provjeri da nismo gotovi
        if fitness == len(clauses):
            return variables
        
        post = update_post(clauses, post, percentage_up, percentage_down)
        
        neighbours = pi_function(variables)
        fitnesses = []
        for neighbour in neighbours:
            neighbour_clauses = calculate_clauses(neighbour, formula)
            fitnesses.append(fitness_function(neighbour_clauses, post, percentage_unit_amount))
        candidate_indices = np.argsort(fitnesses)[-num_of_best:]
        index = np.random.choice(candidate_indices)
        variables = neighbours[index]
        
    print(f'Dosegnut max_iter, najbolje pronađen rezultat: {to_string(variables)}, dobrota: {fitness}/{len(clauses)}')
    return None

#algoritam 4
def GSAT(var_num, formula, max_flips = 100, max_tries = 10000):
    for i in tqdm(range(max_tries)):
        variables = np.random.choice([True, False], size=var_num)
        
        clauses = calculate_clauses(variables, formula)
        fitness = number_of_satisfied(clauses)
        #provjeri da nismo gotovi
        if fitness == len(clauses):
            return variables
        
        for j in range(max_flips):
            neighbours = pi_function(variables)
            fitnesses = []
            for neighbour in neighbours:
                neighbour_clauses = calculate_clauses(neighbour, formula)
                fitnesses.append(number_of_satisfied(neighbour_clauses))
            max_fit = max(fitnesses)

            #izvuci sve kandidate s maksimalnim fitnessom (minimalno nezadovoljenih)
            indices = [i for i, v in enumerate(fitnesses) if v == max_fit]
            candidates = [neighbours[i] for i in indices]
            #izaberi jedan na random
            index = random.randrange(len(candidates))
            variables = candidates[index]
            #provjeri je li to globalni optimum
            if max_fit == len(clauses):
                return variables
    print(f'Dosegnut max_tries, najbolje pronađen rezultat: {to_string(variables)}, dobrota: {fitness}/{len(clauses)}')
    return None
    
#algoritam 5   
def random_walk_SAT(var_num, formula, max_flips = 1000, max_tries = 100000, p=0.5):
    for i in tqdm(range(max_tries)):
        variables = np.random.choice([True, False], size=var_num)
        
        clauses = calculate_clauses(variables, formula)
        fitness = number_of_satisfied(clauses)
        #provjeri da nismo gotovi
        if fitness == len(clauses):
            return variables
        
        for j in range(max_flips):
            if random.random() < p:
                unsatisfied_clauses_indices = [i for i, value in enumerate(clauses) if not value]
                chosen_clause_index = random.choice(unsatisfied_clauses_indices)
                chosen_clause = formula[chosen_clause_index]
                chosen_variable = random.choice(np.abs(chosen_clause))
                variables[chosen_variable-1] = not variables[chosen_variable-1]
            else:
                neighbours = pi_function(variables)
                fitnesses = []
                for neighbour in neighbours:
                    neighbour_clauses = calculate_clauses(neighbour, formula)
                    fitnesses.append(number_of_satisfied(neighbour_clauses))
                max_fit = max(fitnesses)

                #izvuci sve kandidate s maksimalnim fitnessom (minimalno nezadovoljenih)
                indices = [i for i, v in enumerate(fitnesses) if v == max_fit]
                candidates = [neighbours[i] for i in indices]
                #izaberi jedan na random
                index = random.randrange(len(candidates))
                variables = candidates[index]
            
            #izracunaj novi fitness
            clauses = calculate_clauses(variables, formula)
            fitness = number_of_satisfied(clauses)
            #provjeri da nismo gotovi
            if fitness == len(clauses):
                return variables       
    print(f'Dosegnut max_tries, najbolje pronađen rezultat: {to_string(variables)}, dobrota: {fitness}/{len(clauses)}')             
    return None

#algoritam 6
def iterative_search2(var_num, formula, percentage, max_iter=10000):
    variables = np.random.choice([True, False], size=var_num)
    for i in tqdm(range(max_iter)):
        clauses = calculate_clauses(variables, formula)
        fitness = number_of_satisfied(clauses)
        
        #provjeri da nismo već u globalnom optimumu
        if fitness == len(clauses):
            return variables
        
        neighbours = pi_function(variables)
        fitnesses = []
        for neighbour in neighbours:
            neighbour_clauses = calculate_clauses(neighbour, formula)
            fitnesses.append(number_of_satisfied(neighbour_clauses))
        max_fit = max(fitnesses)

        #jesmo li u lokalnom minimumu
        if max_fit < fitness:
            tqdm.write('Lokalni minimum: nasumično mijenjamo dio varijabli')
            #broj varijabli koje mijenjamo
            num_of_flips = int(var_num * percentage)
            indices_to_flip = random.sample(range(var_num), num_of_flips)
            variables[indices_to_flip] = ~variables[indices_to_flip]
            continue

        #izvuci sve kandidate s maksimalnim fitnessom
        indices = [i for i, v in enumerate(fitnesses) if v == max_fit]
        candidates = [neighbours[i] for i in indices]
        #izaberi jedan na random
        index = random.randrange(len(candidates))
        variables = candidates[index]
        #provjeri je li to globalni optimum
        if max_fit == len(clauses):
            return variables
    print(f'Dosegnut max_iter, najbolje pronađen rezultat: {to_string(variables)}, dobrota: {fitness}/{len(clauses)}')
    return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Neispravno pozvan program")
    else:
        algorithm = int(sys.argv[1])
        file_path = sys.argv[2]
        
        var_num, clause_num, formula = read_file(file_path)
    
        match algorithm:
            case 1:
                res = brute_force(var_num, formula)                
            case 2:
                res = iterative_search(var_num, formula)
            case 3:
                res = modified_iter_search(var_num, clause_num, formula)
            case 4:
                res = GSAT(var_num, formula)
            case 5:
                res = random_walk_SAT(var_num, formula)
            case 6:
                res = iterative_search2(var_num, formula, percentage=0.4)
            
        if res is None:
            print('Nezadovoljivo')
        else:
            print(f'Zadovoljivo: {to_string(res)}')