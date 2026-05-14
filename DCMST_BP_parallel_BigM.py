import random

import gurobipy as gp
from gurobipy import GRB
import time
import heapq
import itertools
import concurrent.futures
import multiprocessing

# Assuming these are available in your environment
from Instance_generator import generate_instance
from Gurobi_MTZ import DCMST_gurobi_MTZ

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

def heuristic_dcmst(n, edges, weights, degree_limit, E0=set(), E1=set(), x_bar=None):
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

# ---------------------------------------------------------------------------
# Persistent RMP & Subproblem (UPDATED WITH BIG-M)
# ---------------------------------------------------------------------------
class RMPModel:
    def __init__(self, n, degree_limit):
        self.n   = n
        self.model = gp.Model("RMP")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("Method", 1)
        
        # --- NEW: Big M Penalty ---
        self.M = 1e6 
        
        self.deg_constrs = {
            v: self.model.addConstr(gp.LinExpr() <= degree_limit[v], name=f"deg_{v}")
            for v in range(n)
        }
        
        # --- NEW: Dummy Variable for Guaranteed Feasibility ---
        self.dummy_var = self.model.addVar(lb=0.0, obj=self.M, name="dummy_feasibility")
        
        # Convexity constraint now includes the dummy variable
        self.conv = self.model.addConstr(self.dummy_var == 1, name="conv")
        
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
            
        # --- NEW: Check if the dummy variable is active ---
        # If active, the subproblem failed to generate valid columns to push it out.
        # This node is genuinely mathematically infeasible and must be pruned.
        if self.dummy_var.x > 1e-4:
            return None, None, None, float('inf')
            
        lambda_vals = [v.x for v in self.lam]
        pi  = {v: self.deg_constrs[v].Pi for v in range(self.n)}
        mu  = self.conv.Pi
        
        # --- NEW: Calculate true objective excluding dummy ---
        true_obj = sum(var.Obj * var.x for var in self.lam)
        
        return lambda_vals, pi, mu, true_obj

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
# TOP-LEVEL WORKER FUNCTION
# ---------------------------------------------------------------------------
def process_node(n, edges, weights, degree_limit, E0, E1, valid_cols, current_best_obj):
    """Executes on a separate CPU Core. Returns results to Orchestrator."""
    columns, col_weights, lambda_vals, obj = column_generation(
        n, edges, weights, degree_limit, E0, E1, valid_cols
    )
    
    if columns is None or obj >= current_best_obj - 1e-6:
        return (E0, E1, obj, None, None, columns)
        
    x_bar = {e: sum(lambda_vals[t] for t, col in enumerate(columns) if e in col) for e in edges}
    fractional = [(e, x_bar[e]) for e in edges if 1e-6 < x_bar[e] < 1 - 1e-6]
    
    return (E0, E1, obj, x_bar, fractional, columns)

# ---------------------------------------------------------------------------
# PARALLEL BRANCH AND PRICE ORCHESTRATOR
# ---------------------------------------------------------------------------
def parallel_branch_and_price(n, edges, weights, degree_limit, num_workers=None, time_limit=300):
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
    print(f"Starting Parallel B&P with {num_workers} workers...")
    
    # --- NEW: Safe Root Node Initialization ---
    best_solution, best_obj = heuristic_dcmst(n, edges, weights, degree_limit)
    if best_solution is None:
        best_obj = float('inf')
        fallback_tree = get_feasible_tree(n, edges, weights, set(), set())
        if fallback_tree is None:
            raise ValueError("The underlying graph is disconnected. No solution exists.")
        init_tree = list(fallback_tree)
        print("Root Heuristic failed. Initializing RMP with an unconstrained MST.")
    else:
        init_tree = list(best_solution)
        print(f"Root Heuristic Incumbent: {best_obj:.2f}")
    
    best_lb = float('-inf')

    # Priority Queue for Best-Bound First search
    pq = []
    counter = itertools.count()
    heapq.heappush(pq, (float('-inf'), next(counter), (set(), set(), [set(init_tree)])))
    
    active_futures = set()
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        while pq or active_futures:
            
            # Time-limit check
            if time.time() - start_time >= time_limit:
                best_lb = pq[0][0] if pq else best_lb
                print(f"Time limit reached. UB={best_obj:.2f}  LB={best_lb:.2f}")
                break

            # 1. Fill available CPU cores
            while len(active_futures) < num_workers and pq:
                parent_bound, _, (E0, E1, inherited) = heapq.heappop(pq)
                
                if parent_bound >= best_obj - 1e-6:
                    continue 
                    
                valid_cols = [col for col in inherited if not (col & E0) and E1.issubset(col)]
                if not valid_cols:
                    fallback = get_feasible_tree(n, edges, weights, E0, E1)
                    if fallback is None: continue
                    valid_cols = [fallback]
                
                future = executor.submit(
                    process_node, n, edges, weights, degree_limit, E0, E1, valid_cols, best_obj
                )
                active_futures.add(future)

            if not active_futures: break

            # 2. Wait for FIRST worker
            done, active_futures = concurrent.futures.wait(
                active_futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # 3. Process completed nodes
            for future in done:
                E0, E1, obj, x_bar, fractional, columns = future.result()
                
                if fractional is None: continue 
                
                if not fractional:
                    sol_obj = sum(weights[e] for e in edges if x_bar[e] > 0.5)
                    if sol_obj < best_obj:
                        best_obj = sol_obj
                        best_solution = {e for e in edges if x_bar[e] > 0.5}
                        print(f"*** New Incumbent (Exact): {best_obj:.2f} ***")
                        pq = [(b, c, data) for b, c, data in pq if b < best_obj - 1e-6]
                        heapq.heapify(pq)
                else:
                    heur_tree, heur_obj = heuristic_dcmst(n, edges, weights, degree_limit, E0, E1, x_bar)
                    if heur_obj < best_obj:
                        best_obj = heur_obj
                        best_solution = heur_tree
                        print(f"*** New Incumbent (Heuristic): {best_obj:.2f} ***")
                        pq = [(b, c, data) for b, c, data in pq if b < best_obj - 1e-6]
                        heapq.heapify(pq)
                        
                    e_star = min(fractional, key=lambda p: abs(p[1] - 0.5))[0]
                    heapq.heappush(pq, (obj, next(counter), (E0 | {e_star}, E1.copy(), columns)))
                    heapq.heappush(pq, (obj, next(counter), (E0.copy(), E1 | {e_star}, columns)))

            if pq:
                best_lb = pq[0][0]
            elif not active_futures:
                best_lb = best_obj  

    elapsed = time.time() - start_time
    return best_solution, best_obj, best_lb, elapsed

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n            = 80
    degree_limit = {v: 3 for v in range(n)}
    degree_limit = {v: random.randint(1, 3) for v in range(n)}
    
    W = generate_instance(n, 1, 100, seed=42)

    edges   = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = {(i, j): W[i][j] for (i, j) in edges}

    W_matrix = [[0] * n for _ in range(n)]
    for (i, j), w in weights.items():
        W_matrix[i][j] = w
        W_matrix[j][i] = w

    print("\n--- Gurobi Result ---")
    Edges, best_ub, best_lb, Elapsed_time = DCMST_gurobi_MTZ(n, edges, W_matrix, degree_limit)
    if Edges:
        print("Optimal objective  :", best_ub)
        print("Elapsed time       :", Elapsed_time)

    print("\n--- Parallel Branch-and-Price Result ---")
    bp_sol, bp_obj, bp_lb, bp_time = parallel_branch_and_price(n, edges, weights, degree_limit)
    print("Optimal objective  :", bp_obj)
    print(f"Elapsed time       : {bp_time:.2f} seconds")