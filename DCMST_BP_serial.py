import gurobipy as gp
from gurobipy import GRB
import time
import heapq
import itertools
import concurrent.futures
import multiprocessing
import random as rand

from matplotlib.style import available

from Instance_generator import generate_instance
from DCMST_BP import branch_and_price
from Gurobi_lazycut import DCMST_gurobi_lazy
from Gurobi_MTZ import DCMST_gurobi_MTZ
from Gurobi_integerflow import DCMST_gurobi_integerflow
from Gurobi_multiflow import DCMST_gurobi_multiflow


# ---------------------------------------------------------------------------
# Union-Find / Kruskal Helpers
# ---------------------------------------------------------------------------
def make_uf(n):
    return list(range(n)), [0] * n

def find(parent, u):
    while parent[u] != u:
        parent[u] = parent[parent[u]]   # path compression
        u = parent[u]
    return u

def union(parent, rank, u, v):
    ru, rv = find(parent, u), find(parent, v)
    if ru == rv:
        return False
    if rank[ru] < rank[rv]:
        parent[ru] = rv
    else:
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
    return True

def get_feasible_tree(n, edges, weights, E0, E1):
    """Fallback generator to ensure valid initial columns for restricted nodes."""
    parent, rank = make_uf(n)
    tree = set()
    for e in E1:
        if not union(parent, rank, e[0], e[1]): return None
        tree.add(e)
    available = sorted([e for e in edges if e not in E0 and e not in E1], key=lambda e: weights[e])
    for e in available:
        if len(tree) == n - 1: break
        if union(parent, rank, e[0], e[1]): tree.add(e)
    return tree if len(tree) == n - 1 else None

def find_feasible_tree_mip(n, edges, weights, degree_limit, E0, E1, time_limit=2.0):
    """
    MIP feasibility solver: find any spanning tree that respects degree_limit,
    forces E1, forbids E0. Uses single-commodity integer flow for connectivity.
    Called only when the greedy heuristic fails.
    """
    model = gp.Model("feasibility")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    # Binary edge selection
    x = {e: model.addVar(vtype=GRB.BINARY) for e in edges}

    # Continuous flow on directed arcs (spanning-tree constraint matrix is
    # totally unimodular, so LP relaxation always yields integer solutions).
    f = {}
    for (i, j) in edges:
        f[i, j] = model.addVar(lb=0, ub=n - 1, vtype=GRB.CONTINUOUS)
        f[j, i] = model.addVar(lb=0, ub=n - 1, vtype=GRB.CONTINUOUS)

    # Spanning-tree cardinality
    model.addConstr(gp.quicksum(x[e] for e in edges) == n - 1)

    # Degree constraints
    for v in range(n):
        model.addConstr(
            gp.quicksum(x[e] for e in edges if v in e) <= degree_limit[v]
        )

    # Build neighbour lists for flow conservation
    nbrs = {i: [] for i in range(n)}
    for (u, v) in edges:
        nbrs[u].append(v)
        nbrs[v].append(u)

    # Flow conservation: root supplies n-1, every other node absorbs 1
    for i in range(n):
        rhs = (n - 1) if i == 0 else -1
        model.addConstr(
            gp.quicksum(f[i, j] for j in nbrs[i])
            - gp.quicksum(f[j, i] for j in nbrs[i])
            == rhs
        )

    # Capacity coupling
    for (i, j) in edges:
        model.addConstr(f[i, j] + f[j, i] <= (n - 1) * x[i, j])

    # Fix E1 (must use) and E0 (must not use)
    for e in E1:
        model.addConstr(x[e] == 1)
    for e in E0:
        model.addConstr(x[e] == 0)

    # No objective — pure feasibility
    model.setObjective(gp.quicksum(weights[e] * x[e] for e in edges), GRB.MINIMIZE)

    model.optimize()

    if model.SolCount > 0:
        tree = frozenset(e for e in edges if x[e].X > 0.5)
        return tree
    return None


def LP_heuristic(n, edges, weights, degree_limit, E0=set(), E1=set(), x_bar=None):
    """Greedy LP-guided heuristic for fast upper bounds."""
    parent, rank = make_uf(n)
    tree = set()
    deg = {i: 0 for i in range(n)}
    for u, v in E1:
        if not union(parent, rank, u, v): return None, float('inf')
        tree.add((u, v))
        deg[u] += 1; deg[v] += 1
        if deg[u] > degree_limit[u] or deg[v] > degree_limit[v]: return None, float('inf')
        
    available = [e for e in edges if e not in E0 and e not in E1]
    if x_bar is None:
        available.sort(key=lambda e: weights[e])
    else:
        available.sort(key=lambda e: weights[e] - 10000 * x_bar.get(e, 0))

    for u, v in available:
        if len(tree) == n - 1: break
        if deg[u] < degree_limit[u] and deg[v] < degree_limit[v]:
            if union(parent, rank, u, v):
                tree.add((u, v))
                deg[u] += 1; deg[v] += 1
                
    if len(tree) == n - 1: return tree, sum(weights[e] for e in tree)
    return None, float('inf')

def NearestNeighborhood_heuristic(n, edges, weights, degree_limit, E0=set(), E1=set()):
    """Greedy nearest-neighbor heuristic with Union-Find for connectivity and degree checking."""
    rem = {v: degree_limit[v] for v in degree_limit}
    parent, rank = make_uf(n)
    tree = set()

    # Force E1 edges first; validate degree and acyclicity.
    for u, v in E1:
        if rem[u] <= 0 or rem[v] <= 0:
            return None, float('inf')   # E1 already violates degree limit
        if not union(parent, rank, u, v):
            return None, float('inf')   # E1 creates a cycle → infeasible node
        tree.add((u, v))
        rem[u] -= 1
        rem[v] -= 1

    available = [e for e in edges if e not in E0 and e not in E1]
    available.sort(key=lambda e: (degree_limit[e[0]] + degree_limit[e[1]], weights[e]))

    for u, v in available:
        if len(tree) == n - 1:
            break
        if rem[u] <= 0 or rem[v] <= 0:
            continue
        if not union(parent, rank, u, v):
            continue   # would create a cycle
        tree.add((u, v))
        rem[u] -= 1
        rem[v] -= 1

    if len(tree) == n - 1:
        return tree, sum(weights[e] for e in tree)
    return None, float('inf')

def Feasible_Degree_Tree_Heuristic(n, edges, weights, degree_limit, E0=set(), E1=set()):
    rem = {v: degree_limit[v] for v in degree_limit}
    parent, rank = make_uf(n)
    tree = set()

    # 1. Force E1 edges first
    for u, v in E1:
        if rem[u] <= 0 or rem[v] <= 0:
            return None, float('inf')   
        if not union(parent, rank, u, v):
            return None, float('inf')   
        tree.add((u, v))
        rem[u] -= 1
        rem[v] -= 1

    # 2. Filter available edges
    available = [e for e in edges if e not in E0 and e not in E1]

    # 3. Dynamically pick the most constrained edge
    while len(tree) < n - 1 and available:
        # We want to minimize the remaining capacity (rem[u] + rem[v]). 
        # If remaining capacities are equal, we use weight as a tie-breaker.
        
        best_edge_idx = -1
        best_score = (float('inf'), float('inf')) # (sum_of_rem, weight)

        for i, (u, v) in enumerate(available):
            if rem[u] <= 0 or rem[v] <= 0:
                continue # Skip invalid degrees
            if find(parent, u) == find(parent, v):
                continue # Skip cycles
                
            score = (rem[u] + rem[v], weights[(u, v)])
            if score < best_score:
                best_score = score
                best_edge_idx = i
                
        if best_edge_idx == -1:
            break # No valid edges left to add
            
        # Add the best edge found
        u, v = available.pop(best_edge_idx)
        union(parent, rank, u, v)
        tree.add((u, v))
        rem[u] -= 1
        rem[v] -= 1

    if len(tree) == n - 1:
        return tree, sum(weights[e] for e in tree)
    return None, float('inf')

# ---------------------------------------------------------------------------
# Persistent RMP & Subproblem
# ---------------------------------------------------------------------------
class RMPModel:
    def __init__(self, n, degree_limit):
        self.n   = n
        self.model = gp.Model("RMP")
        self.model.setParam("OutputFlag", 0)
        # self.model.setParam("Method", 1)
        self.deg_constrs = {
            v: self.model.addConstr(gp.LinExpr() <= degree_limit[v], name=f"deg_{v}")
            for v in range(n)
        }
        self.conv = self.model.addConstr(gp.LinExpr() == 1, name="conv")
        self.model.setObjective(gp.LinExpr(), GRB.MINIMIZE)
        self.lam = []

    def add_column(self, col, col_weight):
        deg = {}
        for (u, v) in col:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        coeffs  = [float(d) for d in deg.values()]
        constrs = [self.deg_constrs[v] for v in deg]
        coeffs.append(1.0)
        constrs.append(self.conv)
        gc  = gp.Column(coeffs, constrs)
        var = self.model.addVar(lb=0.0, obj=col_weight, column=gc, name=f"lam_{len(self.lam)}")
        self.lam.append(var)

    def optimize(self):
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            return None, None, None, float('inf')
        lambda_vals = [v.x for v in self.lam]
        pi  = {v: self.deg_constrs[v].Pi for v in range(self.n)}
        mu  = self.conv.Pi
        return lambda_vals, pi, mu, self.model.ObjVal

def solve_subproblem(n, edges, weights, pi, mu, E0, E1):
    parent, rank = make_uf(n)
    tree = []
    reduced_cost = -mu
    for u, v in E1:
        union(parent, rank, u, v)
        tree.append((u, v))
        reduced_cost += weights[(u, v)] - pi[u] - pi[v]

    edges_needed = n - 1 - len(tree)
    candidates = []
    for u, v in edges:
        e = (u, v)
        if e not in E0 and e not in E1:
            w_bar = weights[e] - pi[u] - pi[v]
            candidates.append((w_bar, u, v))
            
    candidates.sort(key=lambda item: item[0])
    
    for w_bar, u, v in candidates:
        if edges_needed == 0: break
        if union(parent, rank, u, v):
            tree.append((u, v))
            reduced_cost += w_bar
            edges_needed -= 1

    if edges_needed > 0: return None, float('inf')
    return tree, reduced_cost

def column_generation(n, edges, weights, degree_limit, E0, E1, init_cols):
    rmp = RMPModel(n, degree_limit)
    columns, col_weights, seen = [], [], set()
    for col in init_cols:
        key = frozenset(col)
        if key not in seen:
            w = sum(weights[e] for e in col)
            rmp.add_column(col, w)
            columns.append(set(col))
            col_weights.append(w)
            seen.add(key)

    while True:
        lambda_vals, pi, mu, obj = rmp.optimize()
        if lambda_vals is None: return None, None, None, float('inf')
        new_tree, rc = solve_subproblem(n, edges, weights, pi, mu, E0, E1)
        if len(columns)%100 == 0:
            print(f"  CG iteration: obj={obj:.2f}  new_tree_rc={rc:.2f}  columns={len(columns)}")
        if new_tree is None or rc >= -1e-6: break
        key = frozenset(new_tree)
        if key not in seen:
            w = sum(weights[e] for e in new_tree)
            rmp.add_column(new_tree, w)
            columns.append(set(new_tree))
            col_weights.append(w)
            seen.add(key)
        else:
            break
    return columns, col_weights, lambda_vals, obj

# ---------------------------------------------------------------------------
# TOP-LEVEL WORKER FUNCTION (Must be global for multiprocessing)
# ---------------------------------------------------------------------------
def process_node(n, edges, weights, degree_limit, E0, E1, valid_cols, current_best_obj):
    """Executes on a separate CPU Core. Returns results to Orchestrator."""
    columns, col_weights, lambda_vals, obj = column_generation(
        n, edges, weights, degree_limit, E0, E1, valid_cols
    )
    
    # Pruned by bound or infeasibility
    if columns is None or obj >= current_best_obj - 1e-6:
        return (E0, E1, obj, None, None, columns)
        
    x_bar = {e: sum(lambda_vals[t] for t, col in enumerate(columns) if e in col) for e in edges}
    fractional = [(e, x_bar[e]) for e in edges if 1e-6 < x_bar[e] < 1 - 1e-6]
    
    return (E0, E1, obj, x_bar, fractional, columns)

# ---------------------------------------------------------------------------
# SERIAL BRANCH AND PRICE
# ---------------------------------------------------------------------------
def branch_and_price_serial(n, edges, weights, degree_limit, time_limit=300):
    print("Starting Serial B&P...")

    start_time = time.time()

    # -----------------------------------------------------------------------
    # Root heuristics — find an initial upper bound
    # -----------------------------------------------------------------------
    best_solution = None
    best_obj = float('inf')

    LP_heuristic_solution, LP_heuristic_obj = LP_heuristic(n, edges, weights, degree_limit)
    nn_solution, nn_objective               = NearestNeighborhood_heuristic(n, edges, weights, degree_limit)
    degreeTree_solution, degreeTree_obj     = Feasible_Degree_Tree_Heuristic(n, edges, weights, degree_limit)

    for sol, obj, label in [
        (LP_heuristic_solution, LP_heuristic_obj, "LP-Guided"),
        (nn_solution,           nn_objective,     "Nearest-Neighbor"),
        (degreeTree_solution,   degreeTree_obj,   "Feasible-Degree-Tree"),
    ]:
        if sol and obj < best_obj:
            best_solution, best_obj = sol, obj
            print(f"Root {label} Heuristic Incumbent: {best_obj:.2f}")

    if best_solution is None:
        mip_tree = find_feasible_tree_mip(
            n, edges, weights, degree_limit, set(), set(), time_limit=min(10.0, time_limit)
        )
        if mip_tree is not None:
            best_solution = mip_tree
            best_obj = sum(weights[e] for e in mip_tree)
            print(f"Root MIP Incumbent: {best_obj:.2f}")
        else:
            print("No feasible tree found at root (problem may be infeasible).")

    init_tree = list(best_solution) if best_solution is not None else []
    best_lb   = float('-inf')

    # -----------------------------------------------------------------------
    # Root column generation (serial, in main process)
    # -----------------------------------------------------------------------
    print("Solving root node...")
    init_cols = [set(init_tree)] if init_tree else []
    root_columns, _, root_lambda, root_obj = column_generation(
        n, edges, weights, degree_limit, set(), set(), init_cols
    )

    pq      = []
    counter = itertools.count()

    if root_columns is None:
        print("Root node infeasible.")
        return best_solution, best_obj, best_lb, time.time() - start_time

    best_lb = root_obj
    print(f"Root LP bound: {root_obj:.2f}")

    root_x_bar    = {e: sum(root_lambda[t] for t, col in enumerate(root_columns) if e in col)
                     for e in edges}
    root_frac     = [(e, root_x_bar[e]) for e in edges if 1e-6 < root_x_bar[e] < 1 - 1e-6]

    if not root_frac:
        sol_obj = sum(weights[e] for e in edges if root_x_bar[e] > 0.5)
        if sol_obj < best_obj:
            best_obj      = sol_obj
            best_solution = {e for e in edges if root_x_bar[e] > 0.5}
        best_lb = best_obj
        print(f"*** Root is integer optimal: {best_obj:.2f} ***")
        return best_solution, best_obj, best_lb, time.time() - start_time

    # Primal heuristic on root LP
    heur_tree, heur_obj = LP_heuristic(n, edges, weights, degree_limit, set(), set(), root_x_bar)
    if heur_obj < best_obj:
        best_obj, best_solution = heur_obj, heur_tree
        print(f"*** New Incumbent (Root Heuristic): {best_obj:.2f} ***")

    # Push root's two children
    e_star = min(root_frac, key=lambda p: abs(p[1] - 0.5))[0]
    heapq.heappush(pq, (root_obj, next(counter), ({e_star}, set(),     root_columns)))
    heapq.heappush(pq, (root_obj, next(counter), (set(),    {e_star},  root_columns)))

    # -----------------------------------------------------------------------
    # Main B&B loop — fully serial
    # -----------------------------------------------------------------------
    while pq:
        print(f"UB: {best_obj:.2f}  |  LB: {best_lb:.2f}  |  Queue: {len(pq)}")

        if time.time() - start_time >= time_limit:
            best_lb = pq[0][0] if pq else best_lb
            print(f"Time limit reached. UB={best_obj:.2f}  LB={best_lb:.2f}")
            return best_solution, best_obj, best_lb, time.time() - start_time

        parent_bound, _, (E0, E1, inherited) = heapq.heappop(pq)

        if parent_bound >= best_obj - 1e-6:
            continue  # prune

        # Prepare initial columns for this node
        valid_cols = [col for col in inherited if not (col & E0) and E1.issubset(col)]
        if not valid_cols:
            fb1, obj1 = LP_heuristic(n, edges, weights, degree_limit, E0, E1)
            fb2, obj2 = NearestNeighborhood_heuristic(n, edges, weights, degree_limit, E0, E1)
            fb3, obj3 = Feasible_Degree_Tree_Heuristic(n, edges, weights, degree_limit, E0, E1)
            fallback  = None
            for fb, obj_fb, label in [(fb1, obj1, "LP"), (fb2, obj2, "NN"), (fb3, obj3, "DegTree")]:
                if fb and (fallback is None or obj_fb < sum(weights[e] for e in fallback)):
                    fallback = fb
                    print(f"  Node fallback ({label}): {obj_fb:.2f}")
            if fallback is None:
                fallback = find_feasible_tree_mip(n, edges, weights, degree_limit, E0, E1)
            if fallback is None:
                continue  # genuinely infeasible — prune
            valid_cols = [fallback]

        # Solve this node (inline — no subprocess)
        E0_r, E1_r, obj, x_bar, fractional, columns = process_node(
            n, edges, weights, degree_limit, E0, E1, valid_cols, best_obj
        )

        if fractional is None:
            continue  # pruned by bound or infeasible

        if not fractional:
            # Integer solution
            sol_obj = sum(weights[e] for e in edges if x_bar[e] > 0.5)
            if sol_obj < best_obj:
                best_obj      = sol_obj
                best_solution = {e for e in edges if x_bar[e] > 0.5}
                print(f"*** New Incumbent (Exact): {best_obj:.2f} ***")
                pq = [(b, c, d) for b, c, d in pq if b < best_obj - 1e-6]
                heapq.heapify(pq)
        else:
            # Primal heuristic
            heur_tree, heur_obj = LP_heuristic(n, edges, weights, degree_limit, E0, E1, x_bar)
            if heur_obj < best_obj:
                best_obj, best_solution = heur_obj, heur_tree
                print(f"*** New Incumbent (Heuristic): {best_obj:.2f} ***")
                pq = [(b, c, d) for b, c, d in pq if b < best_obj - 1e-6]
                heapq.heapify(pq)

            # Branch
            e_star = min(fractional, key=lambda p: abs(p[1] - 0.5))[0]
            heapq.heappush(pq, (obj, next(counter), (E0 | {e_star},  E1.copy(),   columns)))
            heapq.heappush(pq, (obj, next(counter), (E0.copy(),      E1 | {e_star}, columns)))

        # Update LB
        if pq:
            best_lb = pq[0][0]
        else:
            best_lb = best_obj  # proven optimal

    elapsed = time.time() - start_time
    return best_solution, best_obj, best_lb, elapsed

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n            = 300
    seed         = 1
    rand.seed(seed)
    degree_limit = {v: 2 for v in range(n)}
    degree_limit = {v: rand.randint(3,3) for v in range(n)}
    
    # Assuming Instance_generator returns a 2D array W
    W = generate_instance(n, 1, 100, seed=seed)

    edges   = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = {(i, j): W[i][j] for (i, j) in edges}

    # Format matrix for Gurobi
    W_matrix = [[0] * n for _ in range(n)]
    for (i, j), w in weights.items():
        W_matrix[i][j] = w
        W_matrix[j][i] = w

    print("\n--- Parallel Branch-and-Price Result ---")
    bp_sol, bp_obj, bp_lb, bp_time = branch_and_price_serial(n, edges, weights, degree_limit)
    print("Optimal objective  :", bp_obj)
    print(f"Elapsed time       : {bp_time:.2f} seconds")

    # print("\n--- Gurobi Result (MTZ) ---")
    # Edges, best_ub, best_lb, Elapsed_time = DCMST_gurobi_MTZ(n, edges, W_matrix, degree_limit)
    # if Edges:
    #     print("Optimal objective  :", best_ub)
    #     print("Elapsed time       :", Elapsed_time)

    print("\n--- Gurobi Result (Lazy Constraints) ---")
    Edges, best_ub, best_lb, Elapsed_time = DCMST_gurobi_lazy(n, edges, W_matrix, degree_limit)
    if Edges:
        print("Optimal objective  :", best_ub)
        print("Elapsed time       :", Elapsed_time)

    # print("\n--- Gurobi Result (Integer Flow) ---")
    # Edges, best_ub, best_lb, Elapsed_time = DCMST_gurobi_integerflow(n, edges, W_matrix, degree_limit)
    # if Edges:
    #     print("Optimal objective  :", best_ub)
    #     print("Elapsed time       :", Elapsed_time)

    # print("\n--- Gurobi Result (Multi-Flow) ---")
    # Edges, best_ub, best_lb, Elapsed_time = DCMST_gurobi_multiflow(n, edges, W_matrix, degree_limit)
    # if Edges:
    #     print("Optimal objective  :", best_ub)
    #     print("Elapsed time       :", Elapsed_time)