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

def tablemaker():
    for number in range(3,21):
        filename = f"F8_dataset0_{number}.txt"
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                lines = file.readlines()
            
            champion_codes = []
            dimensions = []
            min_distances = []

            first_line = lines[0].strip() # population
            second_line = lines[1].strip() # approximated mindistances
            third_line = lines[2].strip()  # upper bounds
            fourth_line = lines[3].strip() # dimensions
            fifth_line = lines[4].strip() # lower bounds
            sixth_line = lines[5].strip() # a champion code?
            population = ast.literal_eval(first_line) # The codes of this list
            population_dimensions = ast.literal_eval(fourth_line) # The corresponding dimensions of the codes
            championcode_indicators = ast.literal_eval(sixth_line) # Whether the code is champion or not
            for i in range(0,len(population)):
                if championcode_indicators[i] == True:
                    code = population[i]
                    magma_code = vertices2MagmaStr(code)
                    magma_code2 = magma.eval(magma_code)
                    code_dimension = magma.eval('Dimension(LinearCode(generalisedToricMatrix('+magma_code2+')))')
                    mindistance = magma.eval('MinimumWeight(LinearCode(generalisedToricMatrix('+magma_code2+')))')
                    champion_codes.append(code)
                    dimensions.append(code_dimension)
                    min_distances.append(mindistance)
            if len(champion_codes) != 0:
                block_length = [49]*len(champion_codes)
                df = pd.DataFrame({
                    'Vertices': champion_codes,
                    'Block Length': block_length,
                    'Dimension': dimensions,
                    'Minimum Distance': min_distances
                })
                print(df)
                df.to_pickle(f"F8_table_{number}.pkl")

def tablemaker2():
    for number in range(21, 47):
        if number != 30:
            filename = f"F8_datasetnew_{number}.txt"
            if os.path.exists(filename):
                with open(filename, 'r') as file:
                    lines = file.readlines()
                
                champion_codes = []
                dimensions = []
                min_distances = []

                first_line = lines[0].strip() # population
                second_line = lines[1].strip() # approximated mindistances
                third_line = lines[2].strip()  # upper bounds
                fourth_line = lines[3].strip() # dimensions
                fifth_line = lines[4].strip() # lower bounds
                sixth_line = lines[5].strip() # a champion code?
                population = ast.literal_eval(first_line) # The codes of this list
                population_dimensions = ast.literal_eval(fourth_line) # The corresponding dimensions of the codes
                championcode_indicators = ast.literal_eval(sixth_line) # Whether the code is champion or not
                for i in range(0,len(population)):
                    if championcode_indicators[i] == True:
                        code = population[i]
                        magma_code = vertices2MagmaStr(code)
                        magma_code2 = magma.eval(magma_code)
                        code_dimension = magma.eval('Dimension(LinearCode(generalisedToricMatrix('+magma_code2+')))')
                        mindistance = magma.eval('MinimumWeight(LinearCode(generalisedToricMatrix('+magma_code2+')))')
                        champion_codes.append(code)
                        dimensions.append(code_dimension)
                        min_distances.append(mindistance)
                if len(champion_codes) != 0:
                    block_length = [49]*len(champion_codes)
                    df = pd.DataFrame({
                        'Vertices': champion_codes,
                        'Block Length': block_length,
                        'Dimension': dimensions,
                        'Minimum Distance': min_distances
                    })
                    print(df)
                    df.to_pickle(f"F8_table_{number}.pkl")

def tableshower():
    all = pd.DataFrame({
                    'Vertices': [],
                    'Block Length': [],
                    'Dimension': [],
                    'Minimum Distance': []
                })
    for number in (3,4,6,7,8,18, 38, 40, 41, 44, 46):
        df = pd.read_pickle(f"F8_table_{number}.pkl")
        print(df)
        all = pd.concat([all,df])
    print(all)
    latex_table = df.to_latex(index=False)
    print(latex_table)
    df.to_csv(f"F8_table_all.csv")
    df.to_excel("F8_table_all.xlsx")

tableshower()