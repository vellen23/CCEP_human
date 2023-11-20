import os
import numpy as np
import mne
import imageio
import h5py
# import scipy.fftpack
import matplotlib
import pywt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# from scipy import signal
from matplotlib.colors import ListedColormap
import time
import seaborn as sns
# import scipy.io as sio
# from scipy.integrate import simps
import pandas as pd
# from scipy import fft
import matplotlib.mlab as mlab
import sys

sys.path.append('./py_functions')
from scipy.stats import norm
import LL_funcs
from scipy.stats import norm
from tkinter import filedialog
from tkinter import *
import ntpath

root = Tk()
root.withdraw()
import math
import scipy
from scipy import signal
import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import platform
from glob import glob
from scipy.io import savemat
import scipy.cluster.hierarchy as spc
from scipy.spatial import distance
import basic_func as bf
from scipy.integrate import simps
from numpy import trapz
import IO_func as IOF
import BM_func as BMf
import tqdm
from matplotlib.patches import Rectangle
from pathlib import Path
import LL_funcs as LLf
import freq_funcs as ff
from scipy.spatial import distance
#
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft, fftfreq, rfft, irfft, ifft

import scipy.stats as stats
from tqdm.notebook import trange, tqdm
from scipy.signal import find_peaks

method_labels = ['LL', 'Pearson', 'Compound (LL*Pearson)']

def search_sequence_numpy(arr, seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return []  # No match found

def get_phasesync2mean(x_gt, x_trials, tx=1, ty=1, win=0.5, Fs=500):
    # compare all trials to ground truth, define start of response (x, ty)
    # gt: ground truthto be compared, x_trials: single trials (shape (n_trials, time))
    # x_gt = ff.bp_filter(x_gt, 1, 20, Fs)
    x0 = int(tx * Fs)
    x1 = int(x0 + win * Fs)
    y0 = int(ty * Fs)
    y1 = int(y0 + win * Fs)
    if len(x_trials.shape) == 1:
        al1 = np.angle(hilbert(x_gt[x0:x1]), deg=False)
        al2 = np.angle(hilbert(x_trials[y0:y1]), deg=False)
        corr = np.mean(1 - np.sin(np.abs(al1 - al2) / 2))
    else:
        al1 = np.angle(hilbert(x_gt[x0:x1]), deg=False)
        al2 = np.angle(hilbert(x_trials[:, y0:y1]), deg=False)
        corr = np.mean((1 - np.sin(np.abs(al1 - al2) / 2)), -1)
    return corr


def get_pearson2mean_windowed(x_gt, x_trials, tx=1, win=0.25, Fs=500):
    # get pearson coeff for all trials to ground truth for selected time window
    x0 = int(tx * Fs)
    x1 = int(x0 + win * Fs)
    wdp = np.int64(Fs * win)  # 100ms -> 50 sample points
    EEG_pad = np.pad(x_trials, [(0, 0), (np.int64(wdp / 2), np.int64(wdp / 2))], 'constant',
                     constant_values=(0, 0))  # 'reflect'(18, 3006)
    corr_all = np.zeros((x_trials.shape[0], x_trials.shape[1]))
    for i in range(x_trials.shape[1]):  # entire response
        corr_all[:,i]= np.corrcoef(x_gt[x0:x1], EEG_pad[:, i:int(i+(win*Fs))])[0, 1:]
    return corr_all

def get_pearson2mean(x_gt, x_trials, tx=1, ty=1, win=0.25, Fs=500):
    # get pearson coeff for all trials to ground truth for selected time window
    x0 = int(tx * Fs)
    x1 = int(x0 + win * Fs)
    y0 = int(ty * Fs)
    y1 = int(y0 + win * Fs)
    if len(x_trials.shape) == 1:
        corr = np.corrcoef(x_gt[x0:x1], x_trials[y0:y1])[0, 1]
        # corr, p = stats.pearsonr(mn_gt[x0:x1], x_trials[y0:y1],1) # spearmanr
        # corr = corr[0,1]
    else:
        corr = np.corrcoef(x_gt[x0:x1], x_trials[:, y0:y1])[0, 1:]
        # corrs, p = stats.spearmanr(np.expand_dims(x_gt[x0:x1], 0), x_trials[:, y0:y1], 1, alternative='greater')
        # # corr = corr[0,1:]
        # p = p[0, 1:]
        # corr[p > 0.01] = corr[p > 0.01] * 0.8

    return corr


###
def get_trial_Pearson(sc, rc, con_trial_all, EEG_CR, w_p=0.1,w_LL=0.25, Fs=500,t_0=1, t_resp=0,p=1, gt_data =0):
    # for specific connection (stim sc, chan rc) get the pearson coefficients for all trials across blocks
    req = (con_trial_all.Artefact < 1) & (con_trial_all.Chan == rc) & (con_trial_all.Stim == sc)
    StimNum = con_trial_all.loc[req, 'Num'].values.astype('int')
    StimNum_Gt = con_trial_all.loc[req, 'Num'].values.astype('int')
    if len(StimNum_Gt) > 0:
        if np.mean(gt_data) ==0:
            mn = np.nanmean(EEG_CR[rc, StimNum_Gt, :], 0)
        else:
            mn = gt_data[sc,rc,:]
        mn_filt = ff.lp_filter(mn, 30, Fs)
        EEG_LL = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 45, Fs)
        EEG_pearson = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 30, Fs)
        t0 = t_0+t_resp
        if np.isnan(p):
            p=0
        # post stim
        pear1 = get_pearson2mean(mn_filt, EEG_pearson[0], tx=t0, ty=t0, win=w_p, Fs=Fs)  # Pearson# Pearson
        LL_trials = LLf.get_LL_all(EEG_LL[:, :, int(t0 * Fs):int((t0 + 0.3) * Fs)], Fs, w_LL, 0,
                                   np.zeros((len(StimNum), 1)))
        if rc ==53:
            print(rc)

        # todo: easy fix, find better way
        if p:
            for dt in [0.002, 0.004, 0.006, 0.008]:
                pear2 = get_pearson2mean(mn_filt, EEG_pearson[0], tx=t0, ty=t0 - 0.005 + dt, win=w_p,
                                         Fs=500)  # Pearson# Pearson
                pear1 = np.nanmax([pear1, pear2], 0)

        # update contrial list
        con_trial_all.loc[req, 'Pearson'] = pear1
        con_trial_all.loc[req, 'LL0'] = np.max(LL_trials[0],1)
    return con_trial_all

def get_gt_data(sc, rc, EEG_CR,con_trial, gt_data):
    req_gt = (con_trial.Artefact < 1) & (con_trial.Chan == rc) & (con_trial.Stim == sc)  # artefcat == 0
    dat = con_trial[req_gt]
    if len(dat) > 0:
        StimNum = con_trial.loc[req_gt, 'Num'].values.astype('int')
        gt_data[sc, rc,:] = np.nanmean(EEG_CR[rc, StimNum, :], 0)
    return gt_data

def get_surr_connection(sc, rc, EEG_CR, thr_blocks, con_trial, w_LL=0.25, w_p=0.1,
                        Fs=500,t_0=1, t_resp=0,p=1,gt_data =0):  # get_surr_connection(sc, rc,con_trial,EEG_CR,thr_blocks, m=2):

    req = (con_trial.Artefact <1) & (con_trial.Chan == rc) & (con_trial.Stim == sc)
    req_gt = (con_trial.Artefact <1) & (con_trial.Chan == rc) & (con_trial.Stim == sc) # artefcat == 0
    dat = con_trial[req]
    if len(dat) > 0:
        # con_trial = get_trial_Pearson(sc,rc,con_trial,  EEG_CR, t0=1)
        ##ground truth
        if np.mean(gt_data) ==0:
            StimNum = con_trial.loc[req_gt, 'Num'].values.astype('int')
            mn_gt = np.nanmean(EEG_CR[rc, StimNum, :], 0)
        else:
            mn_gt = gt_data[sc,rc,:]
        mn_gt = ff.lp_filter(mn_gt, 30, Fs)
        t0 =t_0 +t_resp
        b_in = 0
        if np.isnan(p):
            p=0
        for b in np.unique(con_trial.Block):
            data_surr = surr_blockwise(sc, rc, con_trial, EEG_CR, mn_gt, b, t0, w_LL=w_LL, w_p=w_p) # ü[LL, Pearson, LL*P]
            data_surr[data_surr[:, 0] > 12, :] = np.nan
            thr = np.nanpercentile(data_surr[:, 2], 95)
            thr_blocks[b_in, sc, rc, 0] = thr
            thr_blocks[b_in, sc, rc, 1] = np.nanpercentile(data_surr[:, 2], 99)
            thr_blocks[b_in, sc, rc, 2] = p
            con_trial.loc[
                req & (con_trial.Block == b) & (con_trial.PLL >= thr), 'Sig'] = 1+1 * p
            con_trial.loc[
                req & (con_trial.Block == b) & (con_trial.PLL < thr), 'Sig'] = 0
            b_in = b_in + 1


    return thr_blocks, con_trial
def update_sig_val(sc, rc, thr_blocks, con_trial):  # get_surr_connection(sc, rc,con_trial,EEG_CR,thr_blocks, m=2):
    # global thr_blocks
    # global con_trial
    req = (con_trial.Artefact <1) & (con_trial.Chan == rc) & (con_trial.Stim == sc)
    dat = con_trial[req]
    if len(dat) > 0:
        b_in = 0
        for b in np.unique(con_trial.Block):
            thr = thr_blocks[b_in, sc, rc, 1] # 0: 95, 1: 99th
            p = thr_blocks[b_in, sc, rc,2] # if sign connection based on mean
            con_trial.loc[
                req & (con_trial.Block == b) & (con_trial.PLL >= thr), 'Sig'] = 1+1 * p
            con_trial.loc[
                req & (con_trial.Block == b) & (con_trial.PLL < thr), 'Sig'] = 0
            b_in = b_in + 1

    return con_trial,thr_blocks

def surr_blockwise(sc, rc, con_trial, EEG_CR, mn_gt, b, t0, w_LL=0.25, w_p=0.1, Fs=500):
    t_surr = np.array([0.2, 1.6, 2.1,0.4, 1.8, 2, 2.3,0])
    # add option: if b==-1, across all blocks
    if len(con_trial.loc[(con_trial.Stim == sc) & (con_trial.Block == b)]) > 0:
        ##surrogates Trials
        mn_trial = np.min(con_trial.loc[(con_trial.Block == b), 'Num'].values.astype('int'))
        mx_trial = np.max(con_trial.loc[(con_trial.Block == b), 'Num'].values.astype('int'))
        real_trials = np.unique(
            con_trial.loc[(con_trial.Stim == sc) & (con_trial.Block == b), 'Num'].values.astype('int'))
        stim_trials = np.unique(con_trial.loc[(con_trial.Stim >= rc - 1) & (con_trial.Stim <= rc + 1) & (
                con_trial.Block == b), 'Num'].values.astype('int'))
        StimNum = np.unique(np.linspace(real_trials - 35, real_trials + 35, 71).flatten())
        StimNum = [i for i in StimNum if i not in stim_trials]
        StimNum = [i for i in StimNum if i not in stim_trials + 1]
        StimNum = [i for i in StimNum if i not in real_trials]
        # StimNum = [i for i in StimNum if i not in real_trials+1]
        StimNum = [i for i in StimNum if i < mx_trial]
        StimNum = [i for i in StimNum if i >= mn_trial]
        StimNum = np.unique(StimNum).astype('int')
        EEG_surr = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 45, Fs)
        EEG_pear = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 30, Fs)
        bad_StimNum = np.where(np.max(abs(EEG_surr[0]), 1) > 1000)
        if (len(bad_StimNum[0]) > 0)&(len(bad_StimNum[0]) < 0.2*len(StimNum)):
            StimNum = np.delete(StimNum, bad_StimNum)
            EEG_surr = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 45, Fs)
            EEG_pear = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 30, Fs)

        # get surrgates data
        n_times = int(np.min([np.ceil(350 / len(StimNum)),len(t_surr)]))
        data_surr = np.zeros((int(len(StimNum) * n_times), 3))
        for i in range(n_times):
            # LL surrogates
            LL_surr = LLf.get_LL_all(EEG_surr[:, :, int((t_surr[i]) * Fs):int((t_surr[i] + 0.3) * Fs)], Fs, w_LL, 0,
                                     np.zeros((len(StimNum), 1)))
            LL_surr = np.max(LL_surr[0], 1)
            # Shape surrogates

            pearson = get_pearson2mean(mn_gt, EEG_pear[0], tx=t0, ty=t_surr[i], win=w_p, Fs=500)  # Pearson
            # phase = get_pearson2mean(mn_gt, EEG_surr[0], tx=t0, ty=t_surr[i],win=0.05, Fs=500) # Pearson
            # phase = get_phasesync2mean(ff.bp_filter(mn_gt, 1, 20, Fs), ff.bp_filter(EEG_surr[0], 1, 20, Fs), tx=t0,
            #                            ty=t_surr[i], win=w_p, Fs=500)  # Phase synchrony# Phase synchrony
            pearson[pearson > 0.9] = pearson[pearson > 0.9]*0.95  # todo: remove later
            data_surr[i * len(StimNum):(i + 1) * len(StimNum), 0] = LL_surr
            data_surr[i * len(StimNum):(i + 1) * len(StimNum), 1] = pearson
            # data_surr[i * len(StimNum):(i + 1) * len(StimNum), 3] = phase
        data_surr[:, 2] = data_surr[:, 1] * data_surr[:, 0]
    else:
        data_surr = np.zeros((100, 3))
        # data_trial = np.zeros((1, 3))
        data_surr[:, :] = np.nan
        # data_trial[:, :] = np.nan
    return data_surr#, data_trial


def get_gt_to(mn_gt, Fs=500):
    x = (mn_gt - np.mean(mn_gt[0:Fs])) / np.std(mn_gt[0:Fs])  # z-score
    N1 = 1000
    for i in [-1, 1]:
        p = 1
        peaks, properties = find_peaks(i * x, height=5, prominence=10, width=[0.02 * Fs, 0.11 * Fs])
        width = properties['widths']
        width = width[(peaks > Fs) & (peaks < 1.3 * Fs)]
        t = properties['left_ips']
        t = t[(peaks > Fs) & (peaks < 1.3 * Fs)]
        peaks = peaks[(peaks > Fs) & (peaks < 1.3 * Fs)]
        if len(t) > 0:
            if t[0] < N1:
                N1 = t[0]  # peaks[0]- width[0]
    if N1 == 1000:
        p = 0
        N1 = Fs
    return N1, p

def get_gt_to_Ph(mn_gt, Fs=500):
    x = (mn_gt - np.mean(mn_gt[0:Fs])) / np.std(mn_gt[0:Fs])  # z-score
    N1 = 1000
    for i in [-1, 1]:
        p = 1
        peaks, properties = find_peaks(i * x, height=4, prominence=1, width=[0.02 * Fs, 0.11 * Fs])
        width = properties['widths']
        width = width[(peaks > Fs) & (peaks < 1.3 * Fs)]
        t = properties['left_ips']
        t = t[(peaks > Fs) & (peaks < 1.3 * Fs)]
        peaks = peaks[(peaks > Fs) & (peaks < 1.3 * Fs)]
        if len(t) > 0:
            if t[0] < N1:
                N1 = t[0]  # peaks[0]- width[0]
    if N1 == 1000:
        p = 0
        N1 = Fs
    return N1, p