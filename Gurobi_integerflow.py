import gurobipy as gp
from gurobipy import GRB
import time


def DCMST_gurobi_integerflow(n, Edges, Weights, Max_degree, time_limit=300):
    """Degree-Constrained Minimum Spanning Tree via Single-Commodity Integer Flow."""
    model = gp.Model("DCMST_IntFlow")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit  = time_limit

    # ------------------------------------------------------------------
    # Edge variables (undirected)
    # ------------------------------------------------------------------
    x = {}
    for i in range(n):
        for j in range(i + 1, n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    def xe(i, j):
        return x[min(i, j), max(i, j)]

    # ------------------------------------------------------------------
    # Integer flow variables on directed arcs
    # ------------------------------------------------------------------
    f = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                f[i, j] = model.addVar(lb=0, ub=n - 1, vtype=GRB.INTEGER, name=f"f_{i}_{j}")

    # ------------------------------------------------------------------
    # Spanning-tree cardinality
    # ------------------------------------------------------------------
    model.addConstr(
        gp.quicksum(x[i, j] for i, j in x) == n - 1,
        name="tree_size"
    )

    # ------------------------------------------------------------------
    # Degree constraints
    # ------------------------------------------------------------------
    for i in range(n):
        model.addConstr(
            gp.quicksum(xe(i, j) for j in range(n) if i != j) <= Max_degree[i],
            name=f"degree_{i}"
        )

    # ------------------------------------------------------------------
    # Flow conservation
    #   node 0 (root) : supply = n-1
    #   node i ≠ 0    : demand = 1
    # ------------------------------------------------------------------
    for i in range(n):
        rhs = (n - 1) if i == 0 else -1
        model.addConstr(
            gp.quicksum(f[i, j] for j in range(n) if j != i)
            - gp.quicksum(f[j, i] for j in range(n) if j != i)
            == rhs,
            name=f"flow_cons_{i}"
        )

    # ------------------------------------------------------------------
    # Capacity coupling: flow only on selected edges
    # ------------------------------------------------------------------
    for i in range(n):
        for j in range(i + 1, n):
            model.addConstr(
                f[i, j] + f[j, i] <= (n - 1) * xe(i, j),
                name=f"cap_{i}_{j}"
            )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    model.setObjective(
        gp.quicksum(Weights[i][j] * x[i, j] for i, j in x),
        GRB.MINIMIZE
    )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    start_time = time.time()
    model.optimize()
    elapsed = time.time() - start_time

    best_lb = model.ObjBound   # always available after optimize()
    best_ub = float('inf')

    if model.SolCount > 0:
        best_ub   = model.ObjVal
        sol_edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
        return sol_edges, best_ub, best_lb, elapsed
    else:
        return None, best_ub, best_lb, elapsed
