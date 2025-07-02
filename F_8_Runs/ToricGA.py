import numpy as np
from MLmodel_F8 import MLModel
from SageFunctions_F8 import SageFunctions, list2vertices, set_seed
import pygad
from GAPopulationFuncs_F8 import generateInitialPopulation, mutation_func, crossover_func, population2populationVertices, reduce_population, vertices2MagmaStr
from sage.all import *
import ast
import sklearn
from sklearn.metrics import *
import pandas as pd
import os
from ToricFunctions import analyse_datasets, explorer
SAGE_EXTCODE = SAGE_ENV['SAGE_EXTCODE']
magma.attach('%s/magma/sage/basic.m'%SAGE_EXTCODE)

# Transformer ML Model parameters
PATH = "transformer_436_vector_post_train_best.pt"
args = None

# Initialize class which provides access to model
model = MLModel(PATH, model_type = "Transformer_v43")


prime = magma.eval(Integer(8))
allvertices = magma.eval('[<x,y> : x,y in [0..'+prime+'-2]]')
print(allvertices)
generalised_toric_matrix = magma.eval('primitive := PrimitiveElement(FiniteField('+ prime +')); generalisedToricMatrix := function(vertices); M := KMatrixSpace(FiniteField('+prime+'), #vertices, ('+prime+'-1)^2); rows := [primitive^(('+allvertices+'[j][1])*vertices[i][1] + ('+allvertices+'[j][2])*vertices[i][2]): j in [1..('+prime+'-1)^2], i in [1..#vertices]]; toricmatrix := M ! rows; return toricmatrix; end function;')

total_runs = 0

if total_runs == 0:
    empty = [set() for i in range(100)]
    for number in range(30,31): # change range to explore certain dimensions
        explorer(number, total_runs, empty)
else:
    all_found, approximate_distances, lower_distances, found_distances = analyse_datasets(total_runs)
    while True:
        for number in range(3,46):
            explorer(number, total_runs, all_found)
        all_found, approximate_distances, lower_distances, found_distances = analyse_datasets(total_runs)    
        total_runs += 1