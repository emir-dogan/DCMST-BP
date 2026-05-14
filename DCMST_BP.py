import gurobipy as gp
from gurobipy import GRB
import time

from Instance_generator import generate_instance
from DCMST_ColumnGeneration import column_generation
from Gurobi_MTZ import DCMST_gurobi_MTZ

def make_uf(n):
    parent = list(range(n))
    rank   = [0] * n
    return parent, rank

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

# ---------------------------------------------------------------------------
# Branch-and-Price
# ---------------------------------------------------------------------------

def branch_and_price(n, edges, weights, degree_limit):
    """
    Branch-and-Price for the Degree-Constrained Minimum Spanning Tree.
    Returns (best_tree_edges, best_objective_value).
    """
    # Build an initial column: TSP-like Nearest Neighbor Hamiltonian cycle,
    # then remove the heaviest arc to get a degree-2 path (spanning tree).
    def nn_hamiltonian_cycle(n, weights):
        """Nearest-neighbor tour starting from node 0. Returns list of edges."""
        visited = [False] * n
        tour    = [0]
        visited[0] = True
        for _ in range(n - 1):
            u    = tour[-1]
            best_v, best_w = -1, float('inf')
            for v in range(n):
                if not visited[v]:
                    e = (min(u, v), max(u, v))
                    if weights[e] < best_w:
                        best_w, best_v = weights[e], v
            tour.append(best_v)
            visited[best_v] = True
        return tour  # implicit closing edge: tour[-1] -> tour[0]

    tour      = nn_hamiltonian_cycle(n, weights)
    # Build the n cycle edges (including closing edge)
    cycle_edges = []
    for k in range(n):
        u = tour[k]
        v = tour[(k + 1) % n]
        cycle_edges.append((min(u, v), max(u, v)))

    # Remove the heaviest edge to get a Hamiltonian path (spanning tree, degree <= 2)
    max_edge = max(cycle_edges, key=lambda e: weights[e])
    init_tree = [e for e in cycle_edges if e != max_edge]

    best_solution = set(init_tree)
    best_obj      = sum(weights[e] for e in init_tree)
    print(f"Heuristic incumbent: {best_obj:.2f}")

    # DFS stack: (E0, E1, inherited_columns, parent_lp_bound)
    # The root node has an assumed parent bound of -infinity
    stack = [(set(), set(), [set(init_tree)], float('-inf'))]
    start_time = time.time()
    
    while stack:
        # The Global Lower Bound in DFS is the minimum parent bound currently in the stack
        # (This is just for printing/monitoring, NOT for pruning)
        global_lower_bound = min((node[3] for node in stack), default=float('-inf'))
        
        print(f"Upper bound: {best_obj:.2f}  |  Global LP bound: {global_lower_bound:.2f}")
        print(f"Stack size: {len(stack)}")
        
        E0, E1, inherited, parent_bound = stack.pop()

        # Prune immediately if the parent's bound is already worse than the incumbent
        # (This saves us from running column generation if the incumbent improved elsewhere)
        if parent_bound >= best_obj - 1e-6:
            continue

        valid_cols = [
            col for col in inherited
            if not (col & E0) and E1.issubset(col)
        ]
        if not valid_cols:
            valid_cols = [set(init_tree)]
        
        node_start_time = time.time()
        columns, col_weights, lambda_vals, obj = column_generation(
            n, edges, weights, degree_limit, E0, E1, valid_cols
        )
        node_elapsed = time.time() - node_start_time
        print(f"  Node processed in {node_elapsed:.2f} seconds. LP bound: {obj:.2f}")

        # PRUNE CONDITION: Infeasible OR LP bound worse than Incumbent Upper Bound
        if columns is None or obj >= best_obj - 1e-6:
            continue
        
        # Aggregate edge values
        x_bar = {
            e: sum(lambda_vals[t] for t, col in enumerate(columns) if e in col)
            for e in edges
        }

        fractional = [(e, x_bar[e]) for e in edges if 1e-6 < x_bar[e] < 1 - 1e-6]

        if not fractional:
            # We found an integer solution
            sol     = {e for e in edges if x_bar[e] > 0.5}
            sol_obj = sum(weights[e] for e in sol)
            if sol_obj < best_obj:
                best_obj      = sol_obj
                best_solution = sol
                print(f"*** New incumbent: {best_obj:.2f} ***")
            # We do NOT branch on integer nodes, so we just let it finish.
        else:
            # Branch on the most fractional edge (closest to 0.5)
            e_star = min(fractional, key=lambda p: abs(p[1] - 0.5))[0]
            
            # Add child nodes to stack, passing 'obj' as their parent_bound
            # Left child: e_star forbidden
            stack.append((E0 | {e_star}, E1.copy(), columns, obj))
            # Right child: e_star fixed
            stack.append((E0.copy(), E1 | {e_star}, columns, obj))

    elapsed_time= time.time()- start_time
    solution = best_solution, best_obj, global_lower_bound, elapsed_time
    return solution


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n            = 100
    degree_limit = {v: 3 for v in range(n)}
    W            = generate_instance(n, 1, 100, seed=42)

    edges   = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = {(i, j): W[i][j] for (i, j) in edges}

    W_matrix = [[0] * n for _ in range(n)]
    for (i, j), w in weights.items():
        W_matrix[i][j] = w
        W_matrix[j][i] = w
    solution = DCMST_gurobi_MTZ(n, edges, W_matrix, degree_limit)
    print("\n--- Gurobi Result ---")
    print("Optimal DCMST edges:", sorted(solution[0]) if solution else None)
    print("Optimal objective  :", solution[1])
    print("Elapsed time       :", solution[2])


    print(f"Running Branch-and-Price on a {n}-node DCMST (degree limit = {degree_limit[0]})...\n")
    solution = branch_and_price(n, edges, weights, degree_limit)

    print("\n--- Result ---")
    print("Optimal DCMST edges:", sorted(solution[0]) if solution else None)
    print("Optimal objective  :", solution[1])
    print("Elapsed time       :", solution[2])


