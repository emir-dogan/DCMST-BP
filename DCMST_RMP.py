import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------------------------------
# Persistent Restricted Master Problem (re-optimized after each column add)
# ---------------------------------------------------------------------------

class RMPModel:
    """
    Holds a live Gurobi LP for the RMP.
    Columns are added one at a time and the model is re-optimized from the
    previous basis (warm start), avoiding a full rebuild each iteration.
    """
    def __init__(self, n, degree_limit):
        self.n   = n
        self.model = gp.Model("RMP")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("Method", 1)  # Dual simplex — best for re-optimization

        # Degree constraints (empty initially, columns added later)
        self.deg_constrs = {
            v: self.model.addConstr(
                gp.LinExpr() <= degree_limit[v], name=f"deg_{v}"
            )
            for v in range(n)
        }
        # Convexity constraint
        self.conv = self.model.addConstr(
            gp.LinExpr() == 1, name="conv"
        )
        self.model.setObjective(gp.LinExpr(), GRB.MINIMIZE)
        self.lam = []   # list of Gurobi Var objects

    def add_column(self, col, col_weight):
        """Add a new lambda variable for the given tree column."""
        # Compute degree of each vertex in O(|col|) instead of O(n * |col|)
        deg = {}
        for (u, v) in col:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1

        coeffs  = [float(d) for d in deg.values()]
        constrs = [self.deg_constrs[v] for v in deg]
        coeffs.append(1.0)
        constrs.append(self.conv)

        gc  = gp.Column(coeffs, constrs)
        var = self.model.addVar(lb=0.0, obj=col_weight, column=gc,
                                name=f"lam_{len(self.lam)}")
        self.lam.append(var)

    def optimize(self):
        """Re-optimize and return (lambda_vals, pi, mu, obj)."""
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            return None, None, None, float('inf')
        lambda_vals = [v.x for v in self.lam]
        pi  = {v: self.deg_constrs[v].Pi for v in range(self.n)}
        mu  = self.conv.Pi
        return lambda_vals, pi, mu, self.model.ObjVal

    def get_lambda_vals(self):
        return [v.x for v in self.lam]

