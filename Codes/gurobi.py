import gurobipy as gp

# Create a new model
model = gp.Model("linear_optimization_example")

# Define the decision variables
x1 = model.addVar(vtype=gp.GRB.CONTINUOUS, name="x1")
x2 = model.addVar(vtype=gp.GRB.CONTINUOUS, name="x2")

# Set objective function: Maximize 2x1 + 3x2
model.setObjective(2 * x1 + 3 * x2, gp.GRB.MAXIMIZE)

# Add constraints
model.addConstr(3 * x1 + 2 * x2 <= 12, "constraint1")
model.addConstr(x1 + 2 * x2 <= 6, "constraint2")

# Optimize the model
model.optimize()

if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found")
    print("x1 =", x1.x)
    print("x2 =", x2.x)
    print("Optimal objective value =", model.objVal)
else:
    print("No solution found")

# You can also retrieve the shadow prices (dual values) for the constraints:
print("Dual values (shadow prices):")
for constr in model.getConstrs():
    print(f"{constr.ConstrName}: {constr.Pi}")

# You can retrieve the reduced costs for the variables (for sensitivity analysis):
print("Reduced costs:")
for var in model.getVars():
    print(f"{var.VarName}: {var.RC}")

# You can also export the model to a file for inspection or further analysis:
model.write("linear_optimization_example.lp")
