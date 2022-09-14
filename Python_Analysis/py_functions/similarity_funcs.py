import numpy as np
import pywt
from scipy import signal
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from scipy.spatial import distance

def dba_cluster(X, n=2):
    km_dba      = TimeSeriesKMeans(n_clusters=n, metric="dtw", max_iter=10, max_iter_barycenter=10, random_state=0).fit(X)
    cc          = km_dba.cluster_centers_
    y           = km_dba.predict(X)
    dist = km_dba.transform(X)
    return cc, y, dist

def euc_dtw(x, y, w=np.inf):
    # create DTW matrix filled with infinity, except for D[0,0]
    # w       = np.inf
    D       = np.ones((len(x) + 1, len(y) + 1)) * np.inf
    D[0, 0] = 0
    for i in range(1, D.shape[0]):       # through each row
        j_values = np.arange(np.max([1,i-w]),np.min([i+w+1,D.shape[1]])).astype('int') # |i-j| < w +1

        for j in j_values: # only columns that meet the requirements, less runtime
            dist    = distance.euclidean(x[i-1], y[j-1]) # 0th element in time series is 1st element in DTW
            D[i, j] = dist + min([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
    return D[len(x), len(y)].astype('float')