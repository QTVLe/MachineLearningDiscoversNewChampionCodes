import numpy as np
from MLmodel_F8 import MLModel
from SageFunctions_F8 import SageFunctions, list2vertices
import pygad
from GAPopulationFuncs_F8 import generateInitialPopulation, mutation_func, crossover_func, population2populationVertices, reduce_population, vertices2MagmaStr
from sage.all import *
import ast
import sklearn
from sklearn.metrics import *
import pandas as pd
import os

SAGE_EXTCODE = SAGE_ENV['SAGE_EXTCODE']
magma.attach('%s/magma/sage/basic.m'%SAGE_EXTCODE)

prime = magma.eval(Integer(8))
allvertices = magma.eval('[<x,y> : x,y in [0..'+prime+'-2]]')
print(allvertices)
generalised_toric_matrix = magma.eval('primitive := PrimitiveElement(FiniteField('+ prime +')); generalisedToricMatrix := function(vertices); M := KMatrixSpace(FiniteField('+prime+'), #vertices, ('+prime+'-1)^2); rows := [primitive^(('+allvertices+'[j][1])*vertices[i][1] + ('+allvertices+'[j][2])*vertices[i][2]): j in [1..('+prime+'-1)^2], i in [1..#vertices]]; toricmatrix := M ! rows; return toricmatrix; end function;')

SF = SageFunctions(8)

best_known_bounds = [49,43,42,40,38,36,35,34,31,30,29,28,27,26,24,24, 23, 21, 21, 20, 19, 18, 17, 16, 16, 15, 14, 14, 13, 12, 12, 11, 11, 10, 9, 8, 8, 7, 7, 6, 6, 6, 5, 4, 4, 3, 2, 2, 1]

def mutation_wrapper(offspring_, ga_instance_):
   return mutation_func(SF, offspring_, ga_instance_)

def generator(num_vertices, target_dimension, all_found):
    
    # Initialize genetic algorithm class

    # public parameters
    num_generations = 200
    num_parents_mating = 200
    sol_per_pop = 300 # Number of solutions in the population.
    parent_selection_type = "sus" #  The parent selection type:
                                  # sss (for steady-state selection),
                                  # rws (for roulette wheel selection),
                                  # sus (for stochastic universal selection),
                                  # rank (for rank selection),
                                  # random (for random selection),
                                  # tournament (for tournament selection)
    keep_elitism = 30 # number of elitism (i.e. best solutions) to keep in the next generation

    # private parameters
    num_genes = 2 * num_vertices
    init_range_low = 0
    init_range_high = 8 - 2
    gene_space = np.arange(init_range_low, init_range_high + 1) # +1 because range upper bound not inclusive
    initial_population = generateInitialPopulation(SF, sol_per_pop, num_genes)

    # Transformer ML model
    PATH = "transformer_436_vector_post_train_best.pt"

    # Initialize class which provides access to model
    model = MLModel(PATH, model_type="Transformer_v43")

    # need to redefine the fitness function
    offset = 300.
    def fitness_func(ga_instance, vertices_list, solution_idx):
        '''should receive "The instance of the pygad.GA class", "solution", "The indices of the solutions in the population"'''
        vertices = SF.list2vertices(vertices_list)
        if tuple(vertices) in all_found[target_dimension]:
            return 10
        else:
            toric_Generator = SF.vertices2GeneratorMatrix(vertices)
            dim = toric_Generator.shape[0]
            mindist = model.predict(toric_Generator)
            return offset + mindist - abs(dim - target_dimension) * 10

    ga_instance = pygad.GA(num_generations=num_generations,
                          num_parents_mating=num_parents_mating,
                          fitness_func=fitness_func,
                          gene_type=int,
                          sol_per_pop=sol_per_pop,
                          num_genes=num_genes,
                          initial_population = initial_population,
                          gene_space = gene_space,
                          parent_selection_type=parent_selection_type,
                          keep_elitism = keep_elitism,
                          mutation_type=mutation_wrapper,
                          crossover_type=crossover_func
                          )
    ga_instance.run()

    population = ga_instance.population
    populationVertices = population2populationVertices(population)

    # Then can run function to eliminate repetitions, returns list of the populationVertices type
    populationVertices = reduce_population(SF, populationVertices, target_dimension)

    population_mindists_predicted = []
    for individual in populationVertices:
        toric_Generator = SF.vertices2GeneratorMatrix(individual)
        population_mindists_predicted.append(float(model.predict(toric_Generator))) # make sure to change input type - generator or generator dual

    return populationVertices, population_mindists_predicted

def analyse_datasets(total_runs): # to analyse the whole dataset
    all_found = [set() for _ in range(49)] # all found codes so far
    approximate_distances = [] # all approximate distances
    lower_distances = [] # all lower bounds of distances
    all_dimensions = [] # all dimensions of corresponding codes
    for i in range(0,total_runs+1):
            for number in range(3, 46):
                filename = f"F8_dataset{i}_{number}.txt"
                if os.path.exists(filename):
                    with open(filename, 'r') as file:
                        lines = file.readlines()
                    first_line = lines[0].strip() #population
                    second_line = lines[1].strip() # approximated mindistances
                    third_line = lines[2].strip()  #upper bounds
                    fourth_line = lines[3].strip() #dimensions
                    fifth_line = lines[4].strip() # lower bounds
                    sixth_line = lines[5].strip() # a champion code?
                    population = set(ast.literal_eval(first_line)) # The codes of this list
                    population_approx_mindists = ast.literal_eval(second_line) # The predicted minimum distances of the population
                    population_lower_mindists = ast.literal_eval(fifth_line) # The calculated lower bounds for min distances of the population
                    population_dimensions = ast.literal_eval(fourth_line) # The dimensions of the population
                    all_found[number].update(population)
                    approximate_distances.extend(population_approx_mindists)
                    lower_distances.extend(population_lower_mindists)
                    all_dimensions.extend(population_dimensions)
    max_per_dim = {}
    for d, dist in zip(all_dimensions, lower_distances):
        # Update the maximum distance for the dimension d
        if d not in max_per_dim or dist > max_per_dim[d]:
            max_per_dim[d] = dist

    # Convert the dictionary into a sorted list of tuples
    result = sorted(max_per_dim.items())
    print(result)

    dimensions, found_distances = zip(*result)

    df = pd.DataFrame({
        'dimension': dimensions,
        'highest found': found_distances
    })

    print(df)

    return all_found, approximate_distances, lower_distances, found_distances

bool_mapping = {"true": True, "false": False} #for boolean values from magma to python

def explorer(number, total_runs, all_found):
    with open(f"F8_datasetnew_{number}.txt", "w") as file:
        population, population_mindists = generator(number, number, all_found)
        file.write(str(population))
        file.write("\n")
        file.write(str(population_mindists))
        file.write("\n")
        lower_bounds = []
        upper_bounds = []
        champion_code = []
        dimensions = []
        for code in population:
            magma_code = vertices2MagmaStr(code)
            magma_code2 = magma.eval(magma_code)
            code_dimension = magma.eval('Dimension(LinearCode(generalisedToricMatrix('+magma_code2+')))')
            dimensions.append(int(code_dimension))
            upper_bound = best_known_bounds[int(code_dimension)-1]
            upper_bounds.append(int(upper_bound))
            verified_mindistance = magma.eval('SetNthreads(90); VerifyMinimumDistanceLowerBound(LinearCode(generalisedToricMatrix('+magma_code2+')),'+str(upper_bound)+')')
            print(verified_mindistance)
            numbers_str = verified_mindistance.split()
            verified_lower = bool_mapping.get(str(numbers_str[0]).lower(),None)            #if verified_lower is false, that means that the upper bound of the code had already reached below the lower bound of the champion distance   
            best_available_lower_bound = int(numbers_str[1])
            #verified_actual = bool_mapping.get(str(numbers_str[2]).lower(),None)
            lower_bounds.append(best_available_lower_bound)
            champion_code.append(verified_lower)
        file.write(str(upper_bounds))
        file.write("\n")
        file.write(str(dimensions))
        file.write("\n")
        file.write(str(lower_bounds))
        file.write("\n")
        file.write(str(champion_code))