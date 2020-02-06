from pulp import *

problem = LpProblem(LpMaximize)

x_1 = LpVariable('x_1', lowBound=0)
x_2 = LpVariable('x_2', lowBound=0)