import os
import sys
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import random
from scipy.stats import entropy

from sklearn.cluster import KMeans
import multiprocessing
import os
import datetime
import numpy as np
import pandas as pd
import nimfa
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, cophenet


def calculate_cophenetic_corr(A):
    """
    Compute the cophenetic correlation coefficient for matrix A.

    Parameters:
    - A : numpy.ndarray
        Input matrix.

    Returns:
    - float
        Cophenetic correlation coefficient.

        The cophenetic correlation coefficient is measure which indicates the dispersion of the consensus matrix and is based on the average of connectivity matrices.
        It measures the stability of the clusters obtained from NMF. It is computed as the Pearson correlation of two distance matrices:
        the first is the distance between samples induced by the consensus matrix; the second is the distance between samples induced by the linkage used in the reordering of the consensus matrix [Brunet2004].

    """

    # Extract the values from the lower triangle of A
    avec = np.array([A[i, j] for i in range(A.shape[0] - 1)
                     for j in range(i + 1, A.shape[1])])

    # Consensus entries are similarities, conversion to distances
    # 1. matrix: distance between samples indced by consensus matrix
    Y = 1 - avec

    # Hierarchical clustering
    # 2. matrix: distance between samples induced by the linkage used in the reordering of the consensus matrix
    Z = linkage(Y, method='average')

    # Cophenetic correlation coefficient of a hierarchical clustering
    coph = cophenet(Z, Y)[0]

    return coph


def nmf_run(args):
    data_matrix, rank, n_runs, target_clusters = args
    consensus = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
    obj = np.zeros(n_runs)
    connectivity_matrices = []
    lowest_obj = float('inf')
    best_H = None
    best_W = None

    for n in range(n_runs):
        nmf = nimfa.Nmf(data_matrix.T, rank=rank, seed="random_vcol", max_iter=10)
        fit = nmf()
        # Matrix of shared membership Cij=1 iff sample i & j are in same cluster (highest H coeff)
        connectivity = fit.fit.connectivity()
        connectivity_matrices.append(connectivity)
        consensus += connectivity
        obj[n] = fit.fit.final_obj  # Final value (of the last performed iteration) of the objective function
        if obj[n] < lowest_obj:
            lowest_obj = obj[n]
            best_H = fit.coef()
            best_W = fit.basis()

    consensus /= n_runs
    coph = calculate_cophenetic_corr(consensus) # perfect consensus matrix, coph = 1
    instability = 1 - coph

    # Computing ARI if target_clusters is provided
    ari = None
    if target_clusters is not None:
        clusters = np.array([np.argmax(best_H[:, i]) for i in range(best_H.shape[1])])
        ari = adjusted_rand_score(target_clusters, clusters)

    # Storing metrics
    metrics = {
        "Rank": rank,
        "Min Final Obj": lowest_obj,
        "Adjusted Rand Index": ari,
        "Cophenetic Correlation": coph,
        "Instability index": instability
    }

    return metrics, consensus, connectivity_matrices, best_H, best_W


def parallel_nmf_consensus_clustering(data_matrix, rank_range, n_runs, experiment_dir=None, target_clusters=None):
    # Create a directory for the experiment
    if experiment_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        experiment_dir = os.path.join("Experiment_" + timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # Using all available cores
    n_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(nmf_run, [(data_matrix, rank, n_runs, target_clusters) for rank in
                                     range(rank_range[0], rank_range[1] + 1)])

    # Saving results
    for (metrics, consensus, connectivity_matrices, best_H, best_W), rank in zip(results, range(rank_range[0],
                                                                                                rank_range[1] + 1)):
        # Creating directory structure for rank
        rank_dir = os.path.join(experiment_dir, f"k={rank}")
        os.makedirs(rank_dir, exist_ok=True)

        connectivity_dir = os.path.join(rank_dir, "connectivity_matrices")
        os.makedirs(connectivity_dir, exist_ok=True)

        # Saving connectivity matrices
        for idx, matrix in enumerate(connectivity_matrices):
            connectivity_path = os.path.join(connectivity_dir, f"connectivity_{idx + 1}.csv")
            np.savetxt(connectivity_path, matrix, delimiter=",")

        # Saving consensus matrix
        consensus_path = os.path.join(rank_dir, "consensus_matrix.csv")
        np.savetxt(consensus_path, consensus, delimiter=",")

        # Saving H_best and W_best matrices
        h_best_path = os.path.join(rank_dir, "H_best.csv")
        np.savetxt(h_best_path, best_H, delimiter=",")

        w_best_path = os.path.join(rank_dir, "W_best.csv")
        np.savetxt(w_best_path, best_W, delimiter=",")

    # Saving metrics as CSV
    metrics_df = pd.DataFrame([res[0] for res in results])
    metrics_path = os.path.join(experiment_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    return experiment_dir  # return the directory where results are saved


def get_clusters(W):
    """
    Assign channels to clusters based on high W values.

    Parameters:
    - W: Basis matrix from NMF decomposition

    Returns:
    - column_cluster_assignments: A dictionary mapping each cluster to its member channels
    """
    column_cluster_assignments = {}
    num_cluster = W.shape[1]

    for cluster_idx in range(num_cluster):
        W_values = W[:, cluster_idx].reshape(-1, 1)  # Reshape the values for clustering
        kmeans = KMeans(n_clusters=2).fit(W_values)
        higher_value_cluster = np.argmax(kmeans.cluster_centers_)

        for channel, assignment in enumerate(kmeans.labels_):
            if assignment == higher_value_cluster:
                column_cluster_assignments.setdefault(cluster_idx, []).append(channel)

    return column_cluster_assignments


def assgin_cluster(V, H):
    # Compute the correlation between each row in V and each row in H
    # Reshape V and H for broadcasting
    V_reshaped = V[:, :, None]
    H_reshaped = H.T[None, :, :]

    # Compute the dot products and the norms
    dot_products = np.sum(V_reshaped * H_reshaped, axis=1)
    V_norms = np.linalg.norm(V, axis=1)[:, None]
    H_norms = np.linalg.norm(H, axis=1)

    # Compute the correlation coefficients
    correlations = dot_products / (V_norms * H_norms)

    # Find the index of the row in H with which each row in V is most similar
    most_similar_rows = np.argmax(correlations, axis=1)

    return most_similar_rows

def hier2_staNMF(X, k_range1, k_range, stab_it=20):
    """
    Run two-level stability NMF clustering on matrix X.

    Parameters:
    - X: Input matrix
    - k_range: Range of ranks to check for stability NMF

    Returns:
    - cluster_structure: Structure storing channels for both clustering levels
    """
    cluster_structure = {}

    # Level 1
    _, instability = stabNMF(X, num_it=stab_it, k0=min(k_range1), k1=max(k_range1), init='nndsvda', it=2000)
    best_k = k_range1[np.argmin(instability)]
    W1, H1 = get_nnmf(X, best_k, init='nndsvda', it=2000)
    clusters_level1 = get_clusters(W1)

    # For each cluster in level 1, run the second level of NMF
    for level1_cluster_idx, channels_in_cluster in clusters_level1.items():
        X_cluster = X[channels_in_cluster]
        cluster_key = 'C' + str(level1_cluster_idx + 1)

        if len(channels_in_cluster) > max(k_range):

            _, instability = stabNMF(X_cluster, num_it=stab_it, k0=min(k_range), k1=max(k_range), init='nndsvda',
                                     it=2000)
            best_k = k_range[np.argmin(instability)]
            W2, H2 = get_nnmf(X_cluster, best_k, init='nndsvda', it=2000)
            clusters_level2 = get_clusters(W2)

            # Map local indices of clusters_level2 to original indices in X
            mapped_clusters_level2 = {}
            for lvl2_idx, local_indices in clusters_level2.items():
                mapped_clusters_level2[cluster_key + '.' + str(lvl2_idx + 1)] = [channels_in_cluster[idx] for idx in
                                                                                 local_indices]

        else:
            # Map local indices of clusters_level2 to original indices in X
            mapped_clusters_level2 = {}
            mapped_clusters_level2[cluster_key + '.1'] = channels_in_cluster
        # Append clusters for both levels to the cluster_structure dictionary
        if cluster_key not in cluster_structure:
            cluster_structure[cluster_key] = {}
        cluster_structure[cluster_key]['level1'] = channels_in_cluster
        cluster_structure[cluster_key]['level2'] = mapped_clusters_level2

    return cluster_structure, W1, H1


def get_entropy(L):
    E = np.array([entropy(row) for row in L])
    A = entropy(E)
    B = np.mean(E)
    return E, A, B


def hier_staNMF(X, k_range, max_clusters, clusters=None, idx=None, parent_entropy=None, cluster_label='',
                threshold=0.1, level=0, stab_it=20):
    if idx is None:
        idx = np.arange(X.shape[0])

    if clusters is None:
        clusters = []

    # Stop if the number of data points is less than the smallest possible number of clusters or if maximum number of clusters has been reached
    if min(X.shape) < min(k_range) or len(clusters) >= max_clusters:
        clusters.append((cluster_label, X, idx))
        return clusters

    # run stability NMF for different ranks
    _, instability = stabNMF(X, num_it=stab_it, k0=min(k_range), k1=max(k_range), init='nndsvda', it=2000)
    best_k = k_range[np.argmin(instability)]

    # Run NMF with best k
    model = NMF(n_components=best_k, init='random', random_state=0)
    W = model.fit_transform(X)

    best_f_values = []
    new_clusters = []  # New list to keep track of the new clusters
    for i in range(best_k):
        # todo: find better way for cluster assignment
        component_idx = np.argmax(W, axis=1) == i
        best_clusters = X[component_idx]
        new_label = cluster_label + 'X' + str(level + 1) + str(i + 1)  # Append the child label to the parent label

        _, A_cluster, B_cluster = get_entropy(best_clusters)
        f_value = A_cluster - B_cluster

        if parent_entropy is None or f_value < parent_entropy:
            best_f_values.append(f_value)
            new_clusters.append((new_label, best_clusters, idx[component_idx]))

    # Only go to next level of recursion if the conditions based on f and g values are met
    for new_cluster in new_clusters:
        new_label, best_clusters, new_idx = new_cluster

        clusters = hier_staNMF(best_clusters, k_range, max_clusters, clusters, new_idx, f_value, new_label,
                               threshold, level + 1, stab_it)

        if len(best_f_values) == 0:  # New condition to handle empty sub_f_values
            clusters.append(new_cluster)
        else:
            g_value = sum(abs(np.array(best_f_values) - np.array(best_f_values)))
            if g_value > threshold:
                clusters.append(new_cluster)

    return clusters


def recursive_stanmf(X, k_range, max_clusters, clusters=None, idx=None, parent_entropy=None, cluster_label='',
                     threshold=0.1, level=0):
    if idx is None:
        idx = np.arange(X.shape[0])

    if clusters is None:
        clusters = []

    # Stop if the number of data points is less than the smallest possible number of clusters or if maximum number of clusters has been reached
    if X.shape[0] < min(k_range) or len(clusters) >= max_clusters:
        clusters.append((cluster_label, X, idx))
        return clusters

    # Apply stability NMF and compute instability
    folderID = "your_folder_"
    model = st.staNMF(X, folderID=folderID, K1=min(k_range), K2=max(k_range), replicates=20, seed=123)
    model.NMF_finished = True
    model.runNMF(spams_nmf(bootstrap=False))
    model.instability("spams_nmf")
    best_k = k_range[np.argmin(model.get_instability())]

    # Run NMF with best k
    model = NMF(n_components=best_k, init='random', random_state=0)
    W = model.fit_transform(X)

    best_f_values = []
    new_clusters = []  # New list to keep track of the new clusters
    for i in range(best_k):
        # todo: find better way for cluster assignment
        component_idx = np.argmax(W, axis=1) == i
        best_clusters = X[component_idx]
        new_label = cluster_label + 'X' + str(level + 1) + str(i + 1)  # Append the child label to the parent label

        _, A_cluster, B_cluster = get_entropy(best_clusters)
        f_value = A_cluster - B_cluster

        if parent_entropy is None or f_value < parent_entropy:
            best_f_values.append(f_value)
            new_clusters.append((new_label, best_clusters, idx[component_idx]))

    # Only go to next level of recursion if the conditions based on f and g values are met
    for new_cluster in new_clusters:
        new_label, best_clusters, new_idx = new_cluster

        clusters = recursive_stanmf(best_clusters, k_range, max_clusters, clusters, new_idx, f_value, new_label,
                                    threshold, level + 1)

        if len(best_f_values) == 0:  # New condition to handle empty sub_f_values
            clusters.append(new_cluster)
        else:
            g_value = sum(abs(np.array(best_f_values) - np.array(best_f_values)))
            if g_value > threshold:
                clusters.append(new_cluster)

    return clusters


def get_nnmf(X, rank, init='nndsvda', it=2000):
    """Non-negative matrix factorization, remove zero rows before computation."""
    W = np.zeros((X.shape[0], rank))
    zero_rows = np.where(X.mean(axis=1) == 0)[0]
    nonzero_rows = np.where(X.mean(axis=1) > 0)[0]
    X0 = np.delete(X, zero_rows, 0)

    model = NMF(n_components=rank, init=init, max_iter=it)
    W0 = model.fit_transform(X0)
    H = model.components_
    W[nonzero_rows, :] = W0

    return W, H


def get_nnmf_forced(m, rank, H0, W0, it=500):
    """Non-negative matrix factorization with initial guess, remove zero rows before computation."""
    W = np.zeros((m.shape[0], rank))
    zero_rows = np.where(m.mean(axis=1) == 0)[0]
    nonzero_rows = np.where(m.mean(axis=1) > 0)[0]
    X0 = np.delete(m, zero_rows, 0)
    W0 = np.delete(W0, zero_rows, 0)
    model = NMF(n_components=rank, init='custom', max_iter=it)
    W0 = model.fit_transform(X0, H=H0, W=W0)
    H = model.components_
    W[nonzero_rows, :] = W0
    return W, H


def get_W_corr(Wa, Wb):
    """Construct n by k matrix of Pearson product-moment correlation coefficients for every combination of two columns in A and B"""
    return np.corrcoef(Wa.T, Wb.T)


def max_corr(corr):
    """Get the mean of the absolute maximum correlation coefficient for each row."""
    return np.abs(corr).max(axis=0).mean()


def amariMaxError(correlation):
    """
    Computes what Wu et al. (2016) described as a 'amari-type error'
    based on average distance between factorization solutions.
    """
    maxCol = np.abs(correlation).max(axis=0)
    colTemp = (1 - maxCol).mean()
    maxRow = np.abs(correlation).max(axis=1)
    rowTemp = (1 - maxRow).mean()
    return (rowTemp + colTemp) / 2


def stabNMF(M_input, num_it=100, k0=2, k1=10, init='nndsvda', it=2000):
    d = M_input.shape[0]  # number of features
    stability = np.zeros((k1 - k0 + 1,))
    instability = np.zeros((k1 - k0 + 1,))

    for k_num, k in enumerate(range(k0, k1 + 1)):  # for each rank
        print(str(k_num) + '/' + str(k1 - k0 + 1), end="\r")
        # for each rank value
        W_all = np.zeros((num_it, d, k))
        for n in range(num_it):
            W, H = get_nnmf(M_input, k)
            W_all[n, :, :] = W

        for i in range(num_it):
            for j in range(i, num_it):
                x = W_all[i]
                y = W_all[j]
                CORR = get_W_corr(x, y)
                simMat_ij = max_corr(CORR) if i != j else 0  # amariMaxError(CORR) if i == j else max_corr(CORR)
                distMat_ij = amariMaxError(CORR)
                stability[k_num] += simMat_ij / (num_it * (num_it - 1) / 2)
                instability[k_num] += distMat_ij / (num_it * (num_it - 1))

    return stability, instability
