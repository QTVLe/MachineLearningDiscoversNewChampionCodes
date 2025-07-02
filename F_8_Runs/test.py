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
#primitive = magma.eval('PrimitiveElement(FiniteField(' + prime +'))')
#print(primitive)
allvertices = magma.eval('[<x,y> : x,y in [0..'+prime+'-2]]')
print(allvertices)
generalised_toric_matrix = magma.eval('primitive := PrimitiveElement(FiniteField('+ prime +')); generalisedToricMatrix := function(vertices); M := KMatrixSpace(FiniteField('+prime+'), #vertices, ('+prime+'-1)^2); rows := [primitive^(('+allvertices+'[j][1])*vertices[i][1] + ('+allvertices+'[j][2])*vertices[i][2]): j in [1..('+prime+'-1)^2], i in [1..#vertices]]; toricmatrix := M ! rows; return toricmatrix; end function;')

best_known_bounds = [49,43,42,40,38,36,35,34,31,30,29,28,27,26,24,24, 23, 21, 21, 20, 19, 18, 17, 16, 16, 15, 14, 14, 13, 12, 12, 11, 11, 10, 9, 8, 8, 7, 7, 6, 6, 6, 5, 4, 4, 3, 2, 2, 1]

vertices = ((3, 3), (4, 2), (5, 3),(7,7),(8,1),(1,5),(5,5))

magma_code = vertices2MagmaStr(vertices)
print(magma_code)
magma_code2 = magma.eval(magma_code)
print(magma_code2)
code_dimension = magma.eval('Dimension(LinearCode(generalisedToricMatrix('+magma_code2+')))')
print(code_dimension)
print(best_known_bounds[int(code_dimension)-1])
mindistance = magma.eval('MinimumWeight(LinearCode(generalisedToricMatrix('+magma_code2+')) : MaximumTime := 60)')
#mindistance_bounds = magma.eval('MinimumWeightBounds(LinearCode(generalisedToricMatrix('+magma_code2+')))')
#mindistance_bounds2 = magma.eval('MinimumWeight(LinearCode(generalisedToricMatrix('+magma_code2+')) : MaximumTime := 60); MinimumWeightBounds(LinearCode(generalisedToricMatrix('+magma_code2+')))')
print(mindistance)
#print(mindistance_bounds)
#print(mindistance_bounds2)
verified_mindistance = magma.eval('VerifyMinimumDistanceLowerBound(LinearCode(generalisedToricMatrix('+magma_code2+')),'+str(best_known_bounds[int(code_dimension)]-1)+')')
print(verified_mindistance)
numbers_str = verified_mindistance.split()
verified_lower = str(numbers_str[0])
print(verified_lower) #if verified_lower is false, that means that the upper bound of the code had already reached below the lower bound of the champion distance
best_available_lower_bound = int(numbers_str[1])
print(best_available_lower_bound)
verified_actual = str(numbers_str[2])
print(verified_actual)

s = "false"
bool_mapping = {"true": True, "false": False}
bool_val = bool_mapping.get(s.lower(), None)  # Returns None if the string isn't "true" or "false"
print(bool_val)
print(bool_mapping.get("true".lower(), None))

print(bool_mapping.get(verified_lower.lower(),None))