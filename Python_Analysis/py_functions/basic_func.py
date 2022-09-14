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
# import LL_funcs
import tqdm
import platform
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from scipy.spatial import distance
import itertools
from scipy.spatial import distance
import math
import LL_funcs as LLf

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]


def read_mat(filename, dataname):
    try:  # open the score file of the first subfolder
        matfile = h5py.File(filename, 'r')[dataname]
        matfile = matfile[()].T
    except OSError:
        matfile = scipy.io.loadmat(filename)[dataname]
    return matfile


def get_Stim_chans(stimlist, lbls):
    labels_all      = lbls.label.values
    labels_clinic   = lbls.Clinic.values
    labels_region   = lbls.Region.values
    coord_all       = np.array([lbls.x.values,lbls.y.values, lbls.z.values ]).T
    # get stimulation channels directly from stimlist
    StimChanSM = np.unique(stimlist.ChanP)

    ChanN = np.zeros((len(StimChanSM),))
    StimChans = []  # np.zeros((len(stim_chan)))
    StimChansC = []  # np.zeros((len(stim_chan)))
    StimChanIx = []  # np.zeros((len(stim_chan)))
    i = 0
    while i < len(StimChanSM):
        ChanN[i] = np.median(stimlist[stimlist.ChanP == StimChanSM[i]].ChanN)
        if ((np.array(lbls.ChanP_SM.values) == StimChanSM[i]) & (np.array(lbls.ChanN_SM.values) == ChanN[i])).any():
            # StimChans.append(labels_SM[(np.array(labels.chan_num.values)==stim_chan[i,0])][0])
            StimChans.append(labels_all[(np.array(lbls.ChanP_SM.values) == StimChanSM[i]) & (
                        np.array(lbls.ChanN_SM.values) == ChanN[i])][0])
            StimChansC.append(labels_clinic[(np.array(lbls.ChanP_SM.values) == StimChanSM[i]) & (
                        np.array(lbls.ChanN_SM.values) == ChanN[i])][0])
            StimChanIx.append(
                lbls[(np.array(lbls.ChanP_SM.values) == StimChanSM[i]) & (np.array(lbls.ChanN_SM.values) == ChanN[i])][
                    'Num'].values[0] - 1)
            i = i + 1
        else:
            StimChanSM = np.delete(StimChanSM, i, 0)

    stimlist = stimlist[np.isin(stimlist.ChanP, StimChanSM)]

    labels_region[labels_region == 'Temporal'] = 'Basotemporal'
    labels_region[labels_region == 'HIPP '] = 'Mesiotemporal'
    labels_region[labels_region == 'HIPP'] = 'Mesiotemporal'
    labels_region[labels_region == 'Temporal'] = 'Laterotemporal'

    return labels_all, labels_region,labels_clinic,coord_all,StimChans, StimChanSM,StimChansC, StimChanIx, stimlist


def check_inStimChan_C(c_s, sc_s, labels_all):
    rr = np.zeros((len(c_s), len(sc_s)))
    for j in range(len(c_s)):
        c = c_s[j]
        lb = labels_all[c]
        for i in range(len(sc_s)):
            sc = np.int64(sc_s[i])
            stim_lb = labels_all[sc]
            t = '-'
            ix = [pos for pos, char in enumerate(lb) if char == t]
            if len(ix) == 5:
                ix = np.int64(ix[3])
            elif len(ix) > 1:
                ix = np.int64(ix[1])
            else:
                ix = np.int64(ix[0])
            chan1 = lb[:ix]
            chan2 = lb[ix + 1:]
            r = 0
            if stim_lb.find(chan1) != -1:
                rr[j, i] = 1
            elif stim_lb.find(chan2) != -1:
                rr[j, i] = 1

        # print(stim_lb)
    return rr


def check_inStimChan(c, sc_s, labels_all):
    rr = np.zeros((len(sc_s),))
    lb = labels_all[c]
    # print(lb)
    for i in range(len(sc_s)):
        sc = np.int64(sc_s[i])
        stim_lb = labels_all[sc]
        t = '-'
        ix = [pos for pos, char in enumerate(lb) if char == t]
        if len(ix) > 3:
            ix = np.int64(ix[2])
        elif len(ix) > 1:
            ix = np.int64(ix[1])
        else:
            ix = np.int64(ix[0])
        chan1 = lb[:ix]
        chan2 = lb[ix + 1:]
        r = 0
        if stim_lb.find(chan1) != -1:
            rr[i] = 1
        elif stim_lb.find(chan2) != -1:
            rr[i] = 1

        # print(stim_lb)
    return rr


def SM2IX(SM, StimChanNums, StimChanIx):
    # SM: stim channel in SM number
    # StimChanNums: all number of stim channels in SM
    # StimChanIx: all stim channels in all channles environment
    ChanIx = np.zeros_like(SM)
    for i in range(len(SM)):
        ChanIx[i] = StimChanIx[np.where(StimChanNums == SM[i])]
    return ChanIx


def zscore_CCEP(data, t_0=1, Fs=500):
    if len(data.shape) == 1:
        m = np.mean(data[int((t_0 - 0.5) * Fs):int((t_0 - 0.05) * Fs)])
        s = np.std(data[int((t_0 - 0.5) * Fs):int((t_0 - 0.05) * Fs)])
        data = (data - m) / s
    else:
        m = np.mean(data[:, int((t_0 - 0.5) * Fs):int((t_0 - 0.05) * Fs)], -1)
        s = np.std(data[:, int((t_0 - 0.5) * Fs):int((t_0 - 0.05) * Fs)], -1)
        m = m[:, np.newaxis]
        s = s[:, np.newaxis]
        data = (data - m) / s
    return data
