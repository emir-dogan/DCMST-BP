import gurobipy as gp
from gurobipy import GRB
import time
from itertools import combinations


def _find_components(nodes, edges_in_sol):
    """Union-Find to identify connected components among selected edges."""
    parent = {v: v for v in nodes}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        parent[find(u)] = find(v)

    for (i, j) in edges_in_sol:
        union(i, j)

    components = {}
    for v in nodes:
        r = find(v)
        components.setdefault(r, set()).add(v)
    return list(components.values())


def DCMST_gurobi_lazy(n, Edges, Weights, Max_degree, time_limit=300):
    model = gp.Model("DCMST")
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit        = time_limit
    model.Params.OutputFlag       = 0

    # Undirected binary edge variables x[i,j] for i < j
    edge_vars = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_vars[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    def x(i, j):
        return edge_vars[min(i, j), max(i, j)]

    # Exactly n-1 edges (spanning tree)
    model.addConstr(
        gp.quicksum(edge_vars[i, j] for i, j in edge_vars) == n - 1,
        name="tree_edges"
    )

    # Degree constraints
    for i in range(n):
        model.addConstr(
            gp.quicksum(x(i, j) for j in range(n) if i != j) <= Max_degree[i],
            name=f"degree_{i}"
        )

    # Objective
    model.setObjective(
        gp.quicksum(Weights[i][j] * edge_vars[i, j] for i, j in edge_vars),
        GRB.MINIMIZE
    )

    # Cycle elimination via lazy subtour elimination callbacks (DFJ)
    def subtour_callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        vals = model.cbGetSolution(edge_vars)
        selected = [(i, j) for (i, j) in edge_vars if vals[i, j] > 0.5]
        components = _find_components(list(range(n)), selected)
        if len(components) == 1:
            return  # Connected — no subtour
        # Add subtour elimination cut for each disconnected component
        for comp in components:
            if len(comp) < n:
                # Sum of edges within S <= |S| - 1
                model.cbLazy(
                    gp.quicksum(
                        x(i, j)
                        for i in comp for j in comp if i < j
                    ) <= len(comp) - 1
                )

    model.Params.OutputFlag = 0
    start_time = time.time()
    model.optimize(subtour_callback)
    elapsed_time = time.time() - start_time

    best_ub = float('inf')
    best_lb = float('-inf')

    # ObjBound is always available after optimize
    if hasattr(model, 'ObjBound'):
        best_lb = model.ObjBound

    if model.SolCount > 0:
        best_ub = model.ObjVal
        sol_edges = [(i, j) for (i, j) in edge_vars if edge_vars[i, j].X > 0.5]
        return sol_edges, best_ub, best_lb, elapsed_time
    else:
        return None, best_ub, best_lb, elapsed_time
