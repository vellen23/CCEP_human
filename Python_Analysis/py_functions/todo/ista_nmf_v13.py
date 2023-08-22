import staNMF as st
from staNMF.nmf_models import spams_nmf
from sklearn.decomposition import NMF
from scipy.stats import entropy
import numpy as np

# This function will compute the entropy for each row of the input matrix
def get_entropy(L):
    E = np.array([entropy(row) for row in L])
    A = entropy(E)
    B = np.mean(E)
    return E, A, B

def recursive_stanmf(X, k_range, max_clusters, clusters=None, idx=None, parent_entropy=None, cluster_label='', threshold=0.1, level=0):
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
        component_idx = np.argmax(W, axis=1) == i
        best_clusters = X[component_idx]
        new_label = cluster_label + 'X' + str(level+1) + str(i+1)  # Append the child label to the parent label

        _, A_cluster, B_cluster = get_entropy(best_clusters)
        f_value = A_cluster - B_cluster

        if parent_entropy is None or f_value < parent_entropy:
            best_f_values.append(f_value)
            new_clusters.append((new_label, best_clusters, idx[component_idx]))

    # Only go to next level of recursion if the conditions based on f and g values are met
    for new_cluster in new_clusters:
        new_label, best_clusters, new_idx = new_cluster

        clusters = recursive_stanmf(best_clusters, k_range, max_clusters, clusters, new_idx, f_value, new_label, threshold, level+1)
        
        if len(best_f_values) == 0:  # New condition to handle empty sub_f_values
            clusters.append(new_cluster)
        else:
            g_value = sum(abs(np.array(best_f_values) - np.array(best_f_values)))
            if g_value > threshold:
                clusters.append(new_cluster)
                
    return clusters


from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y_true = digits.target

clusters = recursive_stanmf(X, range(2, 5), max_clusters=10)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = np.zeros(X.shape[0])
for i, (_, _, cluster_idx) in enumerate(clusters):
    y_pred[cluster_idx] = i

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', square=True, cmap='Blues')
plt.xlabel('Predicted cluster')
plt.ylabel('True class')
plt.title('Confusion Matrix')
plt.show()

