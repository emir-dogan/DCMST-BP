from Instance_generator import generate_instance
from DCMST_BP import branch_and_price
from Gurobi_lazycut import DCMST_gurobi_lazy
from Gurobi_MTZ import DCMST_gurobi_MTZ
from Gurobi_integerflow import DCMST_gurobi_integerflow
from Gurobi_multiflow import DCMST_gurobi_multiflow
from DCMST_BP_parallel_feasibility import parallel_branch_and_price
import random

# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n            = 90
    degree_limit = {v: 2 for v in range(n)}
    degree_limit = {v: 3 for v in range(n)}
    # degree_limit = {v: random.randint(2, 3) for v in range(n)}
    
    # Ensure generate_instance returns a symmetric matrix or upper triangular format
    W            = generate_instance(n, 1, 100, seed=42)

    edges   = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = {(i, j): W[i][j] for (i, j) in edges}

    W_matrix = [[0] * n for _ in range(n)]
    for (i, j), w in weights.items():
        W_matrix[i][j] = w
        W_matrix[j][i] = w
        

    print(f"\nRunning Branch-and-Price on a {n}-node DCMST (degree limit = {degree_limit[0]})...\n")
    bp_solution, bp_obj, best_lb, bp_time = parallel_branch_and_price(n, edges, weights, degree_limit)

    print("\n--- Branch-and-Price Result ---")
    print("Optimal DCMST edges:", sorted(bp_solution) if bp_solution else None)
    print("Best UB            :", bp_obj)
    print("Best lower bound   :", best_lb)
    print(f"Elapsed time       : {bp_time:.2f} seconds")

    print("\n--- Gurobi Result ---")
    # Make sure this unpacks correctly based on how you formatted Gurobi_MTZ
    solution_gur = DCMST_gurobi_integerflow(n, edges, W_matrix, degree_limit, time_limit=300) 
    if solution_gur:
        print("Optimal DCMST edges:", sorted(solution_gur[0]) if isinstance(solution_gur[0], list) else "Returned successfully")
        print("Optimal objective  :", solution_gur[1])
        print("Elapsed time       :", solution_gur[3])