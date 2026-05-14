import gurobipy as gp
from gurobipy import GRB
import time


def DCMST_gurobi_multiflow(n, Edges, Weights, Max_degree, time_limit=300):
    """Degree-Constrained Minimum Spanning Tree via Multi-Commodity Flow formulation."""
    model = gp.Model("DCMST_MCF")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit  = time_limit

    x = {}
    for i in range(n):
        for j in range(i + 1, n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    def x_func(i, j):
        return x[min(i, j), max(i, j)]

    f = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(1, n):
                f[i, j, k] = model.addVar(lb=0.0, ub=1.0, name=f"f_{i}_{j}_{k}")

    model.addConstr(gp.quicksum(x[i, j] for i, j in x) == n - 1,name="tree_size")

    for i in range(n):
        model.addConstr(gp.quicksum(x_func(i, j) for j in range(n) if i != j) <= Max_degree[i],name=f"degree_{i}")

    for k in range(1, n):
        for i in range(n):
            if i == 0:
                rhs = 1      # source of commodity k
            elif i == k:
                rhs = -1     # sink of commodity k
            else:
                rhs = 0

            model.addConstr(
                gp.quicksum(f[i, j, k] for j in range(n) if j != i)- gp.quicksum(f[j, i, k] for j in range(n) if j != i)== rhs, name=f"flow_cons_{i}_{k}")

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(1, n):
                model.addConstr(f[i, j, k] + f[j, i, k] <= x_func(i, j),name=f"cap_{i}_{j}_{k}")

    model.setObjective(
        gp.quicksum(Weights[i][j] * x_func(i, j) for i, j in x), GRB.MINIMIZE)
    
    start_time = time.time()
    model.optimize()
    elapsed = time.time() - start_time

    best_ub = float('inf')
    best_lb = float('-inf')
    best_lb = model.ObjBound  # always available after optimize()

    if model.SolCount > 0:
        best_ub  = model.ObjVal
        sol_edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
        return sol_edges, best_ub, best_lb, elapsed
    else:
        return None, best_ub, best_lb, elapsed
