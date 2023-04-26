import numpy as np
import matplotlib.pyplot as plt

## 1a)
data = np.load('data_all_button_press_window_events_hg.npy')
print("Task 1a)\n number of electrodes: {}\n number of time samples: {}\n\n".format(data.shape[1], data.shape[0]))


## 1b) Implementing KMeans

def get_KMeans_label(data, CC):
    ## label cluster based on smaller euclidean distance to both CC
    n_clusters = CC.shape[0]
    dist2CC = np.zeros((data.shape[0], n_clusters))
    for i in range(n_clusters):
        dist_euclidean = np.sqrt(np.sum(np.subtract(data, CC[i]) ** 2, axis=1))
        dist2CC[:, i] = dist_euclidean
    # label based on smallest euclidean distance to different cluster centers
    y = np.argmin(dist2CC, axis=1)
    return y


def update_cluster(data, CC0):
    ## input:
    # CC0: current cluster centers (centroids)
    # data: data to cluster

    ## output
    # CC: new cluster centers, y: labels, dist: total squared dist

    ## 1. get current label
    y = get_KMeans_label(data, CC0)
    ## 2. calculate new cluster centers based on new labels (y)
    # CC = np.zeros((CC0.shape[0], data.shape[1]))
    CC = np.copy(CC0)
    for i in np.unique(y):
        CC[i] = np.mean(data[y == i], axis=0)  # Calculate centroids as mean of the cluster

    ## 3. update label
    y = get_KMeans_label(data, CC)
    ## 4. sum of squared distances of each electrode from closest CC
    dist = 0
    for i in np.unique(y):
        dist += np.sum(np.sum(np.subtract(data[y == i], CC[i]) ** 2, axis=1))

    return CC, y, dist


def KMeans_clustering(data, n_clusters, max_it=50):
    data = data.T  # transpose data
    idx = np.random.choice(data.shape[0], n_clusters)  # Step 2 => randomly select 2 points
    CC_init = data[idx]

    CC0, y, dist = update_cluster(data, CC_init)  # First loop started from random points
    for i in range(max_it):
        CC, y, dist = update_cluster(data, CC0)  # Then from centroids previous clustering
        if np.array_equal(np.around(CC, 3), np.around(CC0, 3)):
            break
        else:
            CC0 = CC
    return dist, y, CC


# 1c) Finding optimal number of clusters
k_max = 15
dist_nCenter = np.zeros((k_max,))
for i, k in enumerate(np.arange(1, k_max + 1)):
    dist, y_label, _ = KMeans_clustering(data, k)
    dist_nCenter[i] = dist
plt.figure(figsize=(10, 10))
plt.title('1c) Finding optimal number of clusters')
plt.plot(np.arange(1, k_max + 1), dist_nCenter)
plt.ylabel('Total squared distance')
plt.xlabel('N Clusters')
# best cluster: when closest to origin (0,0) -- knee
ix_cluster = np.argmin((dist_nCenter / np.max(dist_nCenter)) ** 2 + (np.arange(1, k_max + 1) / k_max + 1) ** 2)
plt.plot(ix_cluster + 1, dist_nCenter[ix_cluster], 'ro', markersize=5)
plt.show()

print(
    "Task 1c) The total squared distance is decreasing with increasing number of clusters. \n"
    "We would like to have a balance between low number of clusters and low total squared distance. \n"
    "Therefore, the 'elbow' is a good indicator of total number of clusters. \n"
    "It's the point in the graph (red) which is closest to the origin (0,0) \n \n "
    "I got 4 as the optimnal number of cluster. This could change when re-running since KMeans, \n since this algorithm is depending on the initial condition and does not always give the same result.")

## 1d) Visualizing clustering results
k = 4  # ix_cluster + 1  #
dist, y_label, centroids = KMeans_clustering(data, k)
ylim = 1.2 * np.max(abs(centroids))
plt.figure(figsize=(10, 10))
plt.title('1d) Superimpose cluster centers')
for i in range(k):
    plt.plot(centroids[i], label='CC ' + str(i + 1), linewidth=8)
plt.legend()
plt.ylim([-ylim, ylim])
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(nrows=1, ncols=k, sharex=True, sharey=True, figsize=(5 * k, 5))
plt.suptitle('1d) CC and clusters')
ylim = 1.2 * np.max(abs(data))
for i in range(k):
    data_cluster = data[:, y_label == i]
    for j in range(data_cluster.shape[1]):
        axs[i].plot(data_cluster[:, j], lw=1, alpha=0.8, ls='--')
    axs[i].plot(centroids[i, :], color='k', linewidth=5, label='CC')
    axs[i].set_title('CC ' + str(i + 1))
    axs[i].legend()
plt.tight_layout()
plt.ylim([-ylim, ylim])
plt.xlabel('Sample points')
plt.show()

print('\n\n End HW5')
