"""
experiments.py
Runs 10 instances of DCMST comparing Branch-and-Price vs Gurobi (DFJ).
Each instance has a time limit of 300 seconds.
Results are printed as a formatted table.
"""

from Instance_generator import generate_instance
from DCMST_BP import branch_and_price
from Gurobi_lazycut import DCMST_gurobi_lazy
from Gurobi_MTZ import DCMST_gurobi_MTZ
from Gurobi_integerflow import DCMST_gurobi_integerflow
from Gurobi_multiflow import DCMST_gurobi_multiflow
from DCMST_BP_parallel_feasibility import parallel_branch_and_price

TIME_LIMIT = 300  # seconds per solver per instance

# ---------------------------------------------------------------------------
# 10 experiment configurations: (n, degree_limit, seed)
# ---------------------------------------------------------------------------
n = [60,80,100,120,140,160,180]
n = [180,200,220]
n = [60,80,100]
INSTANCES =[]
for i in n:
    for d in [3]:
        for seed in range(3):  
            INSTANCES.append((i, d, seed))

def gap(ub, lb):
    """Optimality gap in percent."""
    if ub == float('inf') or lb <= 0:
        return float('inf')
    return 100.0 * (ub - lb) / abs(ub)

def run_experiment(inst_id, n, d, seed):
    W      = generate_instance(n, 1, 100, seed=seed)
    edges  = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = {(i, j): W[i][j] for (i, j) in edges}
    degree_limit = {v: d for v in range(n)}

    # Build weight matrix for Gurobi
    W_matrix = [[0] * n for _ in range(n)]
    for (i, j), w in weights.items():
        W_matrix[i][j] = w
        W_matrix[j][i] = w

    print(f"\n{'='*70}")
    print(f"Instance {inst_id:2d} | n={n:3d} | degree={d} | seed={seed}")
    print(f"{'='*70}")

    # --- Branch-and-Price ---
    print("  [B&P] running...")
    bp_sol, bp_ub, bp_lb, bp_time = parallel_branch_and_price(
        n, edges, weights, degree_limit
    )
    bp_gap = gap(bp_ub, bp_lb)

    # --- Gurobi ---
    print("  [Gurobi] running...")
    gur_sol, gur_ub, gur_lb, gur_time = DCMST_gurobi_integerflow(
        n, edges, W_matrix, degree_limit, time_limit=TIME_LIMIT
    )
    gur_gap = gap(gur_ub, gur_lb)

    return {
        "id":       inst_id,
        "n":        n,
        "degree":   d,
        "seed":     seed,
        "bp_ub":    bp_ub,
        "bp_lb":    bp_lb,
        "bp_gap":   bp_gap,
        "bp_time":  bp_time,
        "gur_ub":   gur_ub,
        "gur_lb":   gur_lb,
        "gur_gap":  gur_gap,
        "gur_time": gur_time,
    }

def print_table(results):
    header = (
        f"{'#':>2} {'n':>4} {'d':>2} {'seed':>4} | "
        f"{'BP UB':>9} {'BP LB':>9} {'BP Gap%':>8} {'BP Time':>8} | "
        f"{'GUR UB':>9} {'GUR LB':>9} {'GUR Gap%':>8} {'GUR Time':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        bp_gap_str  = f"{r['bp_gap']:8.2f}" if r['bp_gap'] != float('inf') else "     inf"
        gur_gap_str = f"{r['gur_gap']:8.2f}" if r['gur_gap'] != float('inf') else "     inf"
        bp_ub_str   = f"{r['bp_ub']:9.2f}" if r['bp_ub'] != float('inf') else "      inf"
        gur_ub_str  = f"{r['gur_ub']:9.2f}" if r['gur_ub'] != float('inf') else "      inf"
        print(
            f"{r['id']:>2} {r['n']:>4} {r['degree']:>2} {r['seed']:>4} | "
            f"{bp_ub_str} {r['bp_lb']:>9.2f} {bp_gap_str} {r['bp_time']:>8.2f} | "
            f"{gur_ub_str} {r['gur_lb']:>9.2f} {gur_gap_str} {r['gur_time']:>8.2f}"
        )
    print(sep)

if __name__ == "__main__":
    results = []
    for idx, (n, d, seed) in enumerate(INSTANCES, start=1):
        r = run_experiment(idx, n, d, seed)
        results.append(r)

    print_table(results)
