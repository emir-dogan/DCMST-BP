import random

def generate_instance(num_nodes, min_weight, max_weight, seed=None):
    """Generates a random instance of the DCMST problem."""
    if seed is not None:
        random.seed(seed)  # For reproducibility
    Weights = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = random.randint(min_weight, max_weight)  # Random weight between min_weight and max_weight
            Weights[i][j] = weight
            Weights[j][i] = weight  # Symmetric weights for undirected graph
    return Weights
