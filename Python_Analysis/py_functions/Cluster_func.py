import os
import numpy as np
import mne
import h5py
import scipy.fftpack
import matplotlib
import pywt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.patches import Rectangle
import time
import seaborn as sns
import scipy.io as sio
from scipy.integrate import simps
import pandas as pd
from scipy import fft
import sys
import freq_funcs as ff
import LL_funcs
import tqdm
import platform
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from scipy.spatial import distance
import itertools
import math


def dba_cluster(X, n=2):
    km_dba  = TimeSeriesKMeans(n_clusters=n, metric="dtw", max_iter=10, max_iter_barycenter=10, random_state=0).fit(X)
    cc      = km_dba.cluster_centers_
    y       = km_dba.predict(X)
    dist    = km_dba.transform(X)
    dist_cc = np.max(km_dba.transform(cc))
    return cc, y, dist, dist_cc

def get_cluster_pred(sc, rc, LL_CCEP, EEG_resp):
    lists = LL_CCEP[
        ~np.isnan(LL_CCEP.zLL.values) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc) & (LL_CCEP['Condition'] > 0)]
    conds_trials = lists.Condition.values.astype('int')
    stimNum_all = lists.Num.values.astype('int')
    d = 0.5
    trials = EEG_resp[rc, stimNum_all, :]
    trials_z = scipy.stats.zscore(trials, 1)
    data = trials_z[:, np.int64(1 * Fs):np.int64((1 + d) * Fs)]
    cc, y_pred, dist = dba_cluster(np.expand_dims(data, -1))
    D = dist / np.max(dist)

    pred_loss = np.zeros((2,))
    i = 0
    for cond in np.unique(conds_trials):
        d = 0
        for x, y in itertools.combinations(y_pred[conds_trials == cond], 2):
            d += np.square(x - y)
        pred_loss[i] = d  # np.sqrt(d)
        i = 1 + i
    return pred_loss
