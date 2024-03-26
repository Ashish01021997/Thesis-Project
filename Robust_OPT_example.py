import gurobipy as gp
from gurobipy import GRB

# Define the uncertain parameters
mean_coefficient = [1, 2, 3]
std_dev_coefficient = [0.1, 0.2, 0.3]

# Create a Gurobi model
model = gp.Model("Robust_Optimization")

# Define decision variables
x = model.addVars(3, name="x")

# Define the objective function with uncertainty
obj_expr = gp.LinExpr()
for i in range(3):
    obj_expr += (mean_coefficient[i] + x[i]) ** 2

# Minimize the mean of the objective function
model.setObjective(obj_expr, GRB.MINIMIZE)

# Define uncertainty sets (here, we assume a normal distribution)
for i in range(3):
    model.addConstr(x[i] <= mean_coefficient[i] + 2 * std_dev_coefficient[i])
    model.addConstr(x[i] >= mean_coefficient[i] - 2 * std_dev_coefficient[i])

# Optimize the model
model.optimize()

# Print the optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for i in range(3):
        print(f"x[{i}] = {x[i].X}")

    print(f"Objective value: {model.objVal}")
else:
    print("No solution found")
