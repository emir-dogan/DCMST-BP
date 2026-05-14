import gurobipy as gp
from gurobipy import GRB
import time

from Instance_generator import generate_instance

# Weights = generate_instance(100, 1, 100, 1)
# Max_degree = 2
def DCMST_gurobi_MTZ(n, Edges, Weights, Max_degree,Time_limit=300):
    model = gp.Model("DCMST")
    
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    model.addConstr(gp.quicksum(x[i, j] for i in range(n) for j in range(n)) == n - 1, name="tree_edges")

    # Root node has an in-degree of 0
    # All other nodes have an in-degree of exactly 1
    # MTZ Subtour Elimination (Applied only to non-root nodes)
    u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
    for i in range(n):
        model.addConstr(x[i, i] == 0, name=f"no_self_loop_{i}")
    model.addConstr(gp.quicksum(x[j, 0] for j in range(n)) == 0, name="root_in")
    for i in range(1, n):
        model.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1, name=f"in_{i}")
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"mtz_{i}_{j}")

    # Maximum Degree Constraints
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) + gp.quicksum(x[j, i] for j in range(n)) <= Max_degree[i], name=f"degree_{i}")

    model.setObjective(gp.quicksum(Weights[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    
    model.Params.OutputFlag = 0 
    model.Params.TimeLimit = Time_limit
    # model.setParam("Threads", 1) 
    Start_time= time.time()
    model.optimize()
    Elapsed_time= time.time()- Start_time
    
    if model.status == GRB.OPTIMAL:
        Edges = [(i, j) for i in range(n) for j in range(n) if x[i, j].X > 0.5]
        best_ub = model.ObjVal
        best_lb = model.ObjBound
        return Edges, best_ub, best_lb, Elapsed_time
    else:
        if hasattr(model, 'ObjBound'):
            best_lb = model.ObjBound

        if model.SolCount > 0:
            best_ub = model.ObjVal
            sol_edges = [(i, j) for i in range(n) for j in range(n) if x[i, j].X > 0.5]
            return sol_edges, best_ub, best_lb, Elapsed_time
        else:
            return None, best_ub, best_lb, Elapsed_time
        return None

# solution = DCMST_gurobi(len(Weights), [(i, j) for i in range(len(Weights)) for j in range(i + 1, len(Weights))], Weights, Max_degree)
# print("Optimal solution:", solution[0])
# print("Objective value:", solution[1])
# print("Elapsed time:", solution[2])
    