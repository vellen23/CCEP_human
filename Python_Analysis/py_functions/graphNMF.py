import numpy as np
import networkx as nx
from scipy.special import rel_entr
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def compute_cdf(matrix, bins):
    N = matrix.shape[0]
    values = np.array([matrix[i, j] for i in range(N - 1) for j in range(i + 1, N)])
    counts, _ = np.histogram(values, bins=bins, density=True)
    cdf_vals = np.cumsum(counts) / (N * (N - 1) / 2)
    return cdf_vals + 1e-10 # we have to add a small offset to avoid div0!

def compute_cdf_area(cdf_vals, bin_width):
    return np.sum(cdf_vals[:-1]) * bin_width

def compute_delta_k(areas, cdfs):
    delta_k = np.zeros(len(areas))
    delta_y = np.zeros(len(areas))
    delta_k[0] = areas[0]
    for i in range(1, len(areas)):
        delta_k[i] = (areas[i] - areas[i-1]) / areas[i-1]
        delta_y[i] = sum(rel_entr(cdfs[:, i], cdfs[:, i-1]))
    return delta_k, delta_y

def calculate_statistics(M, rank_range, bins):
    k_min, k_max = rank_range
    bin_width = bins[1] - bins[0]
    
    num_bins = len(bins) - 1
    cdfs = np.zeros((num_bins, k_max - k_min + 1))
    areas = np.zeros(k_max - k_min + 1)

    for i, m in enumerate(M):
        cdf_vals = compute_cdf(m, bins)
        areas[i] = compute_cdf_area(cdf_vals, bin_width)
        cdfs[:, i] = cdf_vals
    
    delta_k, delta_y = compute_delta_k(areas, cdfs)
    k_opt = np.argmax(delta_k) + k_min if delta_k.size > 0 else k_min    

    return areas, delta_k, delta_y, k_opt

def graph_reg_nmf(matrix, k, lambda_reg=1.0, max_iter=50):
    n, m = matrix.shape  # since the matrix is square, n == m
    
    # Initialize U and V matrices randomly
    U = np.abs(np.random.randn(n, k))
    V = np.abs(np.random.randn(m, k))
    
    # Generate Graph (G) and Laplacian (L)
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    
    # Compute In-Degree and Out-Degree matrices
    in_degree = np.array(G.in_degree())
    out_degree = np.array(G.out_degree())
    D_in = np.diag(in_degree[:, 1])
    D_out = np.diag(out_degree[:, 1])
    
    connectivity_matrices = []  # to store connectivity matrices at each iteration
    consensus = np.zeros((n, m))
    
    # Iterative update steps for U and V matrices
    for _ in range(max_iter):
        # Update U
        numerator = np.dot(matrix, V) + lambda_reg * np.dot(D_in, U)
        denominator = np.dot(U, np.dot(V.T, V)) + lambda_reg * np.dot(D_out, U) + 1e-10
        U *= numerator / denominator
        
        # Update V
        numerator = np.dot(matrix.T, U) + lambda_reg * np.dot(D_out, V)
        denominator = np.dot(V, np.dot(U.T, U)) + lambda_reg * np.dot(D_in, V) + 1e-10
        V *= numerator / denominator
        
        # Compute and store the connectivity matrix for the current iteration
        connectivity_matrix = np.dot(U, V.T)
        connectivity_matrices.append(connectivity_matrix)
        
        consensus += connectivity_matrix  # Corrected line


    consensus /= max_iter
    if np.isnan(np.max(U)):
        print('stop')
    if np.isnan(np.max(V)):
        print('stop')
    # Normalize the matrices U and V
    U = normalize(U, axis=0, norm='l1')
    V = normalize(V, axis=0, norm='l1')
    
    return U, V, consensus


def extract_subnetworks(G, U):
    subnetworks = {}

    # Assign each node to the community with which it has the highest association in U
    for node_idx, memberships in enumerate(U):
        community_idx = np.argmax(memberships)

        # Add the node to the corresponding community's subgraph
        if community_idx not in subnetworks:
            subnetworks[community_idx] = []
        subnetworks[community_idx].append(node_idx)

    # Create subgraph for each identified community
    subgraphs = []
    for community_idx, nodes in subnetworks.items():
        subgraph = G.subgraph(nodes)
        subgraphs.append(subgraph)

    return subgraphs


def run(connectivity_matrix, k_max = 3):
    # Initialize a dictionary to store U, V, and consensus matrices for each k
    results = {}

    for k in range(1, k_max):  # Starting from 1 because the minimum number of clusters should be 1
        U, V, consensus_matrix = graph_reg_nmf(connectivity_matrix, k)

        # Store U, V, and consensus matrix for this k value
        results[k] = {"U": U, "V": V, "consensus": consensus_matrix}

    # After this loop, you have U, V, and consensus matrices for each k value in the results dictionary.
    # Now, calculate statistics for each k value and find the optimal k.

    max_delta_k = -np.inf
    optimal_k = None
    optimal_U = None
    optimal_V = None
    optimal_consensus = None

    for k in range(1, k_max):
        bins = np.linspace(0, 1, 101)
        C, delta_k, delta_y, k_opt = calculate_statistics([results[k]["consensus"]], [1, k_max-1], bins)  # range(1, k_max)

        if np.any(delta_k > max_delta_k):
            max_delta_k = delta_k
            optimal_k = k
            optimal_U = results[k]["U"]
            optimal_V = results[k]["V"]
            optimal_consensus = results[k]["consensus"]

    print("Optimal k:", optimal_k)
    print("U for optimal k:\n", optimal_U)
    print("V for optimal k:\n", optimal_V)
    print("Consensus matrix for optimal k:\n", optimal_consensus)

    G = nx.from_numpy_array(connectivity_matrix, create_using=nx.DiGraph)
    subnetworks = extract_subnetworks(G, optimal_U)
    
    # Plot the original graph with subnetworks highlighted in different colors
    plt.figure(figsize=(10,7))
    pos = nx.spring_layout(G, seed=42)  # for consistent layout between plots
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw nodes, color nodes in the same subnetwork with the same color
    colors = plt.cm.tab10.colors  # list of colors
    for idx, subnetwork in enumerate(subnetworks):
        nx.draw_networkx_nodes(G, pos, nodelist=subnetwork.nodes(), node_color=[colors[idx % len(colors)]], label=f"Subnetwork {idx+1}")
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title('Original Graph with Identified Subnetworks')
    plt.legend()
    plt.axis('off')
    plt.show()