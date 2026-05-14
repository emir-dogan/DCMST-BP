# ---------------------------------------------------------------------------
# Subproblem: Modified Kruskal's MST (Optimized)
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

def solve_subproblem(n, edges, weights, pi, mu, E0, E1):
    """
    Solves the pricing subproblem: MST with modified weights
    Highly optimized for inner-loop Column Generation.
    """

    parent, rank = make_uf(n)
    tree = []
    reduced_cost = -mu  # Start with -mu to avoid doing it at the end

    # 1. Force fixed edges first and pre-calculate their reduced cost
    for u, v in E1:
        union(parent, rank, u, v)
        tree.append((u, v))
        reduced_cost += weights[(u, v)] - pi[u] - pi[v]

    # Track how many more edges we need (avoids calling len(tree) in the loop)
    edges_needed = n - 1 - len(tree)

    # 2. Decorate-Sort: Calculate modified weight ONCE per valid edge.
    # We store (w_bar, u, v). Python sorts tuples natively by the first element incredibly fast.
    candidates = []
    for u, v in edges:
        e = (u, v)
        if e not in E0 and e not in E1:
            w_bar = weights[e] - pi[u] - pi[v]
            candidates.append((w_bar, u, v))

    # Sort natively by w_bar (no slow lambda function required)
    candidates.sort(key=lambda item: item[0])

    # 3. Build the rest of the tree and sum reduced cost on the fly
    for w_bar, u, v in candidates:
        if edges_needed == 0:
            break
        if union(parent, rank, u, v):
            tree.append((u, v))
            reduced_cost += w_bar
            edges_needed -= 1

    # If we didn't find enough edges, the graph is disconnected
    if edges_needed > 0:
        return None, float('inf')

    return tree, reduced_cost