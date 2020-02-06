from pulp import *

# problem definition
prob = LpProblem(sense=LpMaximize)

# variables
x_1 = LpVariable('x_1', lowBound=0)
x_2 = LpVariable('x_2', lowBound=0)

# objective function
prob += 3*x_1 + 5*x_2

# constraints
prob += x_1 + 2*x_2 <= 50
prob += 8*x_1 + 3*x_2 <= 240

GLPK().solve(prob)

# solution
for v in prob.variables():
    print(f'{v.name} = {v.varValue}')

print(f'objective = {value(prob.objective)}')
