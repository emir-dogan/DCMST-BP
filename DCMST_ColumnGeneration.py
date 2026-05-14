from DCMST_RMP import RMPModel
from DCMST_SP import solve_subproblem

# ---------------------------------------------------------------------------
# Column Generation  (persistent RMP — re-optimized after each column add)
# ---------------------------------------------------------------------------

def column_generation(n, edges, weights, degree_limit, E0, E1, init_cols):
    """
    Iteratively adds improving columns to the RMP until no negative
    reduced-cost column exists.
    Returns (columns, col_weights, lambda_vals, obj_value).
    """
    # Build persistent RMP and seed with initial columns
    rmp = RMPModel(n, degree_limit)
    columns     = []
    col_weights = []
    seen        = set()
    for col in init_cols:
        key = frozenset(col)
        if key not in seen:
            w = sum(weights[e] for e in col)
            rmp.add_column(col, w)
            columns.append(set(col))
            col_weights.append(w)
            seen.add(key)

    while True:
        # Re-optimize from previous basis (warm start)
        lambda_vals, pi, mu, obj = rmp.optimize()
        if lambda_vals is None:
            return None, None, None, float('inf')
        new_tree, rc = solve_subproblem(n, edges, weights, pi, mu, E0, E1)

        if new_tree is None or rc >= -1e-6:
            break   # no improving column

        key = frozenset(new_tree)
        if key not in seen:
            w = sum(weights[e] for e in new_tree)
            rmp.add_column(new_tree, w)   # add column to live model
            columns.append(set(new_tree))
            col_weights.append(w)
            seen.add(key)
        else:
            # Degenerate pricing can return the same column repeatedly.
            # If the model does not change, re-optimizing loops forever.
            break

    return columns, col_weights, lambda_vals, obj
