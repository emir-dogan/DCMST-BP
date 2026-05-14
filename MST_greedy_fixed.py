def Fixed_Arc_Greedy_MST(arcs, arcs_exists, arcs_violating, Weights):
    """Greedy MST with fixed and violating arcs using Union-Find."""
    n = len(Weights)
    parent = list(range(n))
    rank = [0] * n

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        u_root, v_root = find(u), find(v)
        if u_root == v_root:
            return False
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        else:
            parent[v_root] = u_root
            if rank[u_root] == rank[v_root]:
                rank[u_root] += 1
        return True

    mst = set()
    # Add fixed arcs first
    for u, v in arcs_exists:
        if union(u, v):
            mst.add((u, v))

    # Prepare candidate arcs (exclude violating and already-included arcs)
    candidates = [arc for arc in arcs if arc not in arcs_violating and arc not in mst]
    candidates.sort(key=lambda arc: Weights[arc[0]][arc[1]])

    for u, v in candidates:
        if union(u, v):
            mst.add((u, v))
        if len(mst) == n - 1:
            break

    return mst

# Example usage:
import random

def generate_weights(n, min_weight=1, max_weight=100, seed=None):
    if seed is not None:
        random.seed(seed)
    Weights = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            w = random.randint(min_weight, max_weight)
            Weights[i][j] = w
            Weights[j][i] = w
    return Weights

def generate_arcs(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def pick_random_arcs(arcs, k, exclude=None):
    if exclude is None:
        exclude = set()
    available = [arc for arc in arcs if arc not in exclude]
    return random.sample(available, min(k, len(available)))

# Parameters (customize as needed)
n = 100  # number of nodes
min_weight = 1
max_weight = 400
seed = 1
num_fixed = 20
num_violating = 50

Weights = generate_weights(n, min_weight, max_weight, seed)
arcs = generate_arcs(n)
arcs_exists = pick_random_arcs(arcs, num_fixed)
arcs_violating = pick_random_arcs(arcs, num_violating, exclude=set(arcs_exists))

mst = Fixed_Arc_Greedy_MST(arcs, arcs_exists, arcs_violating, Weights)
print("Constructed MST with fixed arcs:", mst)