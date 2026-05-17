"""
Run_experiment.py
=================
Experimental comparison of:
  1. Gurobi Lazy-Cut (DFJ)
  2. Gurobi Integer-Flow
  3. Gurobi Multi-Commodity Flow
  4. Parallel Branch-and-Price (B&P)

Instance grid:
  n      = [20, 50, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000]
  degree = [2, 3, 4]
  seeds  = 0..9  (10 instances per (n, d) combination)

Sweep order: method -> d -> n -> seed
  Each method is run across all (d, n, seed) up to its early-stop limit,
  then the next method starts.

Early-stopping rule (per method, per degree):
  If ALL seeds for a given n fail (timeout or memory limit), skip all larger n for that (method, d).

Output: results.csv  (one row per solve, written immediately after each solve)
Columns: n, d, seed, method, UB, LB, Gap%, Time(s), Status
"""

import csv
import time
import psutil
import os
import threading
import gc

from Instance_generator import generate_instance
from Gurobi_lazycut      import DCMST_gurobi_lazy
from Gurobi_integerflow  import DCMST_gurobi_integerflow
from Gurobi_multiflow    import DCMST_gurobi_multiflow
from DCMST_BP_parallel_fMIP import parallel_branch_and_price

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIME_LIMIT  = 300
MEMORY_LIMIT_GB = 8  # Memory limit in GB
N_SEEDS     = 10
SKIP_THRESH = N_SEEDS   # skip larger n once ALL seeds fail (timeout or memory)

# Global flag for memory monitoring
memory_exceeded = False

N_VALUES = [20, 50, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000]
D_VALUES = [2, 3, 4, 5]

# TIME_LIMIT  = 10
# N_SEEDS     = 3
# SKIP_THRESH = 2   # skip larger n once ALL seeds time out for a given n

# N_VALUES = [20, 50,80]
# D_VALUES = [2, 3,]

METHODS = [
    ("LazyDFJ", DCMST_gurobi_lazy),
    ("IntFlow", DCMST_gurobi_integerflow),
    ("MCF",     DCMST_gurobi_multiflow),
    ("ParBP",   None),   # handled separately
]

OUTPUT_FILE = "results.csv"
CSV_COLS    = ["n", "d", "seed", "method", "UB", "LB", "Gap%", "Time(s)", "Status"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gap(ub, lb):
    if ub == float('inf') or ub <= 0 or lb <= 0:
        return float('inf')
    return 100.0 * (ub - lb) / abs(ub)


def _fmt(v):
    """Format float for CSV; 'inf' for infinities."""
    if v in (float('inf'), float('-inf')):
        return "inf"
    return f"{v:.4f}"


def get_memory_usage_gb():
    """Get current process memory usage in GB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert bytes to GB


def cleanup_memory():
    """Force garbage collection and pause to allow memory cleanup."""
    gc.collect()  # Force garbage collection
    time.sleep(2)  # Allow OS and Gurobi to reclaim memory


def monitor_memory(stop_event, memory_limit_gb):
    """Monitor memory usage and set flag if limit exceeded."""
    global memory_exceeded
    while not stop_event.is_set():
        current_mem = get_memory_usage_gb()
        if current_mem > memory_limit_gb:
            memory_exceeded = True
            print(f"\n    MEMORY LIMIT EXCEEDED: {current_mem:.2f} GB > {memory_limit_gb} GB")
            return
        time.sleep(0.5)  # Check every 0.5 seconds


def run_gurobi_solver(solver_fn, n, edges, W_matrix, degree_limit):
    global memory_exceeded
    memory_exceeded = False
    
    # Start memory monitoring thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_memory, args=(stop_event, MEMORY_LIMIT_GB))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    initial_mem = get_memory_usage_gb()
    print(f"[mem: {initial_mem:.2f}GB]", end=" ", flush=True)
    
    t0 = time.time()
    try:
        result = solver_fn(n, edges, W_matrix, degree_limit, time_limit=TIME_LIMIT)
    except Exception as exc:
        elapsed = time.time() - t0
        stop_event.set()
        monitor_thread.join(timeout=1)
        print(f"    ERROR: {exc}")
        return float('inf'), float('-inf'), elapsed, True, f"ERROR"
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1)
    
    elapsed = time.time() - t0
    final_mem = get_memory_usage_gb()
    
    # Check if memory limit was exceeded
    if memory_exceeded:
        return float('inf'), float('-inf'), elapsed, False, "MEM_LIMIT"
    
    if result is None:
        ub, lb = float('inf'), float('-inf')
    else:
        _, ub, lb, elapsed = result
        # No feasible solution found but LB may still be valid
        if result[0] is None:
            ub = float('inf')
    timed_out = elapsed >= TIME_LIMIT * 0.99
    return ub, lb, elapsed, timed_out, "TL" if timed_out else "OK"


def run_bp_solver(n, edges, weights, degree_limit):
    global memory_exceeded
    memory_exceeded = False
    
    # Start memory monitoring thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_memory, args=(stop_event, MEMORY_LIMIT_GB))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    initial_mem = get_memory_usage_gb()
    print(f"[mem: {initial_mem:.2f}GB]", end=" ", flush=True)
    
    try:
        _, ub, lb, elapsed = parallel_branch_and_price(
            n, edges, weights, degree_limit, time_limit=TIME_LIMIT
        )
    except Exception as exc:
        stop_event.set()
        monitor_thread.join(timeout=1)
        print(f"    ERROR: {exc}")
        return float('inf'), float('-inf'), TIME_LIMIT, True, "ERROR"
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1)
    
    final_mem = get_memory_usage_gb()
    
    # Check if memory limit was exceeded
    if memory_exceeded:
        return float('inf'), float('-inf'), elapsed, False, "MEM_LIMIT"
    
    timed_out = elapsed >= TIME_LIMIT * 0.99
    return ub, lb, elapsed, timed_out, "TL" if timed_out else "OK"


def _build_instance(n, d, seed):
    W            = generate_instance(n, 1, n*2, seed=seed)
    edges        = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights      = {(i, j): W[i][j] for (i, j) in edges}
    degree_limit = {v: d for v in range(n)}
    W_matrix     = [[0] * n for _ in range(n)]
    for (i, j), w in weights.items():
        W_matrix[i][j] = w
        W_matrix[j][i] = w
    return edges, weights, degree_limit, W_matrix


def _write_row(writer, csvfile, n, d, seed, method_name,
               ub, lb, g, elapsed, status):
    try:
        writer.writerow({
            "n":       n,
            "d":       d,
            "seed":    seed,
            "method":  method_name,
            "UB":      _fmt(ub),
            "LB":      _fmt(lb),
            "Gap%":    _fmt(g),
            "Time(s)": f"{elapsed:.2f}",
            "Status":  status,
        })
        csvfile.flush()
    except Exception as exc:
        print(f"    [WARNING: CSV write failed: {exc}]")


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_all():
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLS)
        writer.writeheader()
        csvfile.flush()

        n_methods = len(METHODS)
        
        for d in D_VALUES:
            print(f"\n{'='*72}")
            print(f"# DEGREE = {d}")
            print(f"{'='*72}")
            
            for m_idx, (method_name, solver_fn) in enumerate(METHODS, 1):
                print(f"\n{'#'*72}")
                print(f"# METHOD [{m_idx}/{n_methods}]: {method_name} | d={d}")
                print(f"{'#'*72}")

                # Per-method early-stop flag and per-n failure counter (timeout or memory)
                skipped_method = False
                failed_count = {n: 0 for n in N_VALUES}

                for n in N_VALUES:
                    for seed in range(N_SEEDS):
                        print(f"\n  [{method_name}]  d={d}  n={n}  seed={seed}",
                              end="  ", flush=True)

                        if skipped_method:
                            print("-> SKIP (early-stop)")
                            _write_row(writer, csvfile, n, d, seed, method_name,
                                       float('inf'), float('-inf'),
                                       float('inf'), TIME_LIMIT, "SKIP")
                            continue

                        # Build instance
                        try:
                            edges, weights, degree_limit, W_matrix = \
                                _build_instance(n, d, seed)
                        except Exception as exc:
                            print(f"-> INST_ERR: {exc}")
                            _write_row(writer, csvfile, n, d, seed, method_name,
                                       float('inf'), float('-inf'),
                                       float('inf'), 0.0, "INST_ERR")
                            continue

                        # Solve
                        if method_name == "ParBP":
                            ub, lb, elapsed, timed_out, status = run_bp_solver(
                                n, edges, weights, degree_limit
                            )
                        else:
                            ub, lb, elapsed, timed_out, status = run_gurobi_solver(
                                solver_fn, n, edges, W_matrix, degree_limit
                            )

                        g = gap(ub, lb)
                        print(f"UB={_fmt(ub)}  LB={_fmt(lb)}  gap={_fmt(g)}%  "
                              f"t={elapsed:.1f}s  [{status}]")

                        _write_row(writer, csvfile, n, d, seed, method_name,
                                   ub, lb, g, elapsed, status)

                        # Clean up memory after each solve
                        cleanup_memory()

                        # Count failures (timeout or memory limit)
                        if status in ("TL", "MEM_LIMIT"):
                            failed_count[n] += 1

                    # After all seeds for this (method, d, n): check early-stop
                    if not skipped_method and failed_count[n] >= SKIP_THRESH:
                        print(f"\n  *** Early-stop: [{method_name}] d={d} — all "
                              f"{SKIP_THRESH} seeds failed (timeout/memory) at n={n}. "
                              "Skipping larger n. ***")
                        skipped_method = True
                
                # Clean up memory after each method completes
                print(f"\n  Cleaning up memory after {method_name}...")
                cleanup_memory()
                current_mem = get_memory_usage_gb()
                print(f"  Memory after cleanup: {current_mem:.2f} GB")

    print(f"\nResults written to  {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_all()
        print("\nAll done.")
    except KeyboardInterrupt:
        print("\nInterrupted. Rows already written to CSV are preserved.")
    except Exception as main_exc:
        print(f"\nFATAL ERROR: {main_exc}")
        import traceback
        traceback.print_exc()

