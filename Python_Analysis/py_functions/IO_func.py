import os
import numpy as np
import mne
import h5py
import scipy.fftpack
import matplotlib
import basic_func as bf
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
import LL_funcs as LLf
import tqdm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from scipy.spatial import distance
import itertools
import math
from numpy import trapz
from glob import glob
import ntpath
import Peaks_funcs as Pkf

# regions         = pd.read_excel("T:\EL_experiment\Patients\\" +'all'+"\elab_labels.xlsx", sheet_name='regions', header=0)
# color_regions   = regions.color.values
# regions         = regions.label.values
cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]

Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3

# dur[0,:]       = np.int32(np.sum(abs(dur)))
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])


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
            if len(ix) > 1:
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
        if len(ix) > 1:
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
def get_LL_all_LTI(EEG_resp, stimlist, lbls, bad_chans, Fs=500,t_0=1,w_LL=0.25):
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM,StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(stimlist,
                                                                                                          lbls)
    data_LL = np.zeros((1, 12))  # RespChan, Int, LL, LLnorm, State
    stim_spec = stimlist[(stimlist.noise <1) ]  # &(stimlist.noise ==0)
    stimNum = stim_spec.Num.values  # [:,0]
    noise_val = stim_spec.noise.values  # [:,0]
    if len(stimNum)>0:
        #resps = EEG_resp[:, stimNum, :]
        resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
        ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))
        IPIs = np.expand_dims(np.array(stim_spec.IPI_ms.values), 1)
        #LL = LLf.get_LL_both(data=resps, Fs=Fs, IPI=IPIs, t_0=1, win=w)
        LL_trial = LLf.get_LL_all(resps[:, :, int(t_0 * Fs):int((t_0+0.5) * Fs)], Fs, w_LL, 1, IPIs)
        LL_peak = np.max(LL_trial, 2)
        t_peak = np.argmax(LL_trial, 2) + int((t_0 - w_LL / 2) * Fs)
        t_peak[t_peak < (t_0 * Fs)] = t_0 * Fs
        inds = np.repeat(np.expand_dims(t_peak, 2), int(w_LL * Fs), 2)
        inds = inds + np.arange(int(w_LL * Fs))
        pN = np.min(np.take_along_axis(resps, inds, axis=2), 2)
        pP = np.max(np.take_along_axis(resps, inds, axis=2), 2)
        p2p = abs(pP - pN)
        for c in range(len(LL_peak)):
            val = np.zeros((LL_peak.shape[1], 12))
            val[:, 0] = c  # response channel
            val[:, 1] = ChanP1
            val[:, 4] = stim_spec.Int_prob.values  # Intensity
            val[:, 3] = noise_val
            val[:, 2] = LL_peak[c, :]  # PP
            val[:, 6] = stim_spec['h'].values
            val[:, 5] = stim_spec['condition'].values
            val[:, 7] = stimNum
            val[:, 8] = stim_spec['StimNum'].values
            val[:, 9] = stim_spec.type.values
            val[:, 10] = stim_spec.stim_block.values
            val[:, 11] =  p2p[c, :]  # LL_peak_ratio[c, :]  # ratio
            # set stimulation channels to nan
            val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 3] = 1
            val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 2] = np.nan
            # if its the recovery channel, check if strange peak is appearing
            pks = np.max(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1)
            pks_loc = np.argmax(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1) + np.int64(
                (t_0 - 0.05) * Fs)
            ix = np.where(
                (pks > 500) & (pks_loc > np.int64((t_0 - 0.005) * Fs)) & (pks_loc < np.int64((t_0 + 0.008) * Fs)))
            # original stim number:
            sn = stim_spec.StimNum.values[ix]
            rec_chan = stimlist.loc[np.isin(stimlist.StimNum, sn - 1), 'ChanP'].values
            rec_chan = SM2IX(rec_chan, StimChanSM, np.array(StimChanIx))
            if np.isin(c, rec_chan):
                val[ix, 3] = 1

            data_LL = np.concatenate((data_LL, val), axis=0)

        data_LL = data_LL[1:-1, :]  # remove first row (dummy row)
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "P2P": data_LL[:, 11], "Artefact": data_LL[:, 3],
             "nLL": data_LL[:, 2], "Int": data_LL[:, 4],
             'Condition': data_LL[:, 5], 'Hour': data_LL[:, 6], "Block": data_LL[:, 10], "Stim_type": data_LL[:, 9],
             "Num": data_LL[:, 7],"Num_block": data_LL[:,8]})

        # distance
        for s in np.unique(LL_all.Stim):
            s = np.int64(s)
            for c in np.unique(LL_all.Chan):
                c = np.int64(c)
                LL_all.loc[(LL_all.Stim == s) & (LL_all.Chan == c), 'd'] = np.round(
                    distance.euclidean(coord_all[s], coord_all[c]), 2)
        # remove bad channels
        LL_all.loc[(LL_all.Chan).isin(bad_chans), 'Artefact'] = 1
        LL_all.loc[(LL_all.Chan).isin(bad_chans), 'Artefact'] = 1

    return LL_all

def get_LL_all_block(EEG_resp, stimlist, lbls, bad_chans, w=0.25, Fs=500,t_0=1,w_LL=0.25):
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM,StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(stimlist,
                                                                                                          lbls)
    data_LL = np.zeros((1, 12))  # RespChan, Int, LL, LLnorm, State
    stim_spec = stimlist[(stimlist.IPI_ms == 0) ]  # &(stimlist.noise ==0)
    stimNum = stim_spec.StimNum.values  # [:,0]
    noise_val = stim_spec.noise.values  # [:,0]
    if len(stimNum)>0:
        #resps = EEG_resp[:, stimNum, :]
        resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
        ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))
        IPIs = np.expand_dims(np.array(stim_spec.IPI_ms.values), 1)
        #LL = LLf.get_LL_both(data=resps, Fs=Fs, IPI=IPIs, t_0=1, win=w)
        LL_trial = LLf.get_LL_all(resps[:, :, int(t_0 * Fs):int((t_0+0.5) * Fs)], Fs, w_LL, 1, IPIs)
        LL_peak = np.max(LL_trial, 2)
        t_peak = np.argmax(LL_trial, 2) + int((t_0 - w_LL / 2) * Fs)
        t_peak[t_peak < (t_0 * Fs)] = t_0 * Fs
        inds = np.repeat(np.expand_dims(t_peak, 2), int(w_LL * Fs), 2)
        inds = inds + np.arange(int(w_LL * Fs))
        pN = np.min(np.take_along_axis(resps, inds, axis=2), 2)
        pP = np.max(np.take_along_axis(resps, inds, axis=2), 2)
        p2p = abs(pP - pN)
        for c in range(len(LL_peak)):
            val = np.zeros((LL_peak.shape[1], 12))
            val[:, 0] = c  # response channel
            val[:, 1] = ChanP1
            val[:, 4] = stim_spec.Int_prob.values  # Intensity
            val[:, 3] = noise_val
            val[:, 2] = LL_peak[c, :]  # PP
            val[:, 6] = stim_spec['h'].values
            val[:, 5] = stim_spec['condition'].values
            val[:, 7] = stimNum
            val[:, 8] = stim_spec.date.values
            val[:, 9] = stim_spec.sleep.values
            val[:, 10] = stim_spec.stim_block.values
            val[:, 11] =  p2p[c, :]  # LL_peak_ratio[c, :]  # ratio
            # set stimulation channels to nan
            val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 3] = 1
            val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 2] = np.nan
            # if its the recovery channel, check if strange peak is appearing
            pks = np.max(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1)
            pks_loc = np.argmax(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1) + np.int64(
                (t_0 - 0.05) * Fs)
            ix = np.where(
                (pks > 500) & (pks_loc > np.int64((t_0 - 0.005) * Fs)) & (pks_loc < np.int64((t_0 + 0.008) * Fs)))
            # original stim number:
            sn = stim_spec.StimNum.values[ix]
            rec_chan = stimlist.loc[np.isin(stimlist.StimNum, sn - 1), 'ChanP'].values
            rec_chan = SM2IX(rec_chan, StimChanSM, np.array(StimChanIx))
            if np.isin(c, rec_chan):
                val[ix, 3] = 1

            data_LL = np.concatenate((data_LL, val), axis=0)

        data_LL = data_LL[1:-1, :]  # remove first row (dummy row)
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "P2P": data_LL[:, 11], "Artefact": data_LL[:, 3],
             "nLL": data_LL[:, 2], "Int": data_LL[:, 4],
             'Condition': data_LL[:, 5], 'Hour': data_LL[:, 6], "Block": data_LL[:, 10], "Sleep": data_LL[:, 9],
             "Num": data_LL[:, 7],"Num_block": data_LL[:, 7], "Date": data_LL[:, 8]})

        # distance
        for s in np.unique(LL_all.Stim):
            s = np.int64(s)
            for c in np.unique(LL_all.Chan):
                c = np.int64(c)
                LL_all.loc[(LL_all.Stim == s) & (LL_all.Chan == c), 'd'] = np.round(
                    distance.euclidean(coord_all[s], coord_all[c]), 2)
        # remove bad channels
        LL_all.loc[(LL_all.Chan).isin(bad_chans), 'Artefact'] = 1
        LL_all.loc[(LL_all.Chan).isin(bad_chans), 'Artefact'] = 1
        # file = path_patient + '/Analysis/InputOutput/' + cond_folder + '/data/con_trial'+block_l+'.csv'
        # LL_all.to_csv(file, index=False, header=True)  # scat_plot = scat_plot.fillna(method='ffill')
        # print(file + ' -- stored')
    else:
        data_LL = np.zeros((1, 12))
        data_LL[:,2:4] = np.nan
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "P2P": data_LL[:, 2], "Noise": data_LL[:, 3],
             "nLL": data_LL[:, 2], "Int": data_LL[:, 4],
             'Condition': data_LL[:, 5], 'Hour': data_LL[:, 6], "Block": data_LL[:, 10], "Sleep": data_LL[:, 9],
             "Num": data_LL[:, 7],"Num_block": data_LL[:, 7], "Date": data_LL[:, 8]})
    return LL_all

def get_LL_all_LTI(EEG_resp, stimlist, lbls, bad_chans, Fs=500,t_0=1,w_LL=0.25):
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM,StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(stimlist,
                                                                                                          lbls)
    data_LL = np.zeros((1, 12))  # RespChan, Int, LL, LLnorm, State
    stim_spec = stimlist[(stimlist.IPI_ms == 0) ]  # &(stimlist.noise ==0)
    stimNum = stim_spec.StimNum.values  # [:,0]
    noise_val = stim_spec.noise.values  # [:,0]
    if len(stimNum)>0:
        #resps = EEG_resp[:, stimNum, :]
        resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
        ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))
        IPIs = np.expand_dims(np.array(stim_spec.IPI_ms.values), 1)
        #LL = LLf.get_LL_both(data=resps, Fs=Fs, IPI=IPIs, t_0=1, win=w)
        LL_trial = LLf.get_LL_all(resps[:, :, int(t_0 * Fs):int((t_0+0.5) * Fs)], Fs, w_LL, 1, IPIs)
        LL_peak = np.max(LL_trial, 2)
        t_peak = np.argmax(LL_trial, 2) + int((t_0 - w_LL / 2) * Fs)
        t_peak[t_peak < (t_0 * Fs)] = t_0 * Fs
        inds = np.repeat(np.expand_dims(t_peak, 2), int(w_LL * Fs), 2)
        inds = inds + np.arange(int(w_LL * Fs))
        pN = np.min(np.take_along_axis(resps, inds, axis=2), 2)
        pP = np.max(np.take_along_axis(resps, inds, axis=2), 2)
        p2p = abs(pP - pN)
        for c in range(len(LL_peak)):
            val = np.zeros((LL_peak.shape[1], 12))
            val[:, 0] = c  # response channel
            val[:, 1] = ChanP1
            val[:, 4] = stim_spec.Int_prob.values  # Intensity
            val[:, 3] = noise_val
            val[:, 2] = LL_peak[c, :]  # PP
            val[:, 6] = stim_spec['h'].values
            val[:, 5] = stim_spec['condition'].values
            val[:, 7] = stimNum
            val[:, 8] = stim_spec.date.values
            val[:, 9] = stim_spec.sleep.values
            val[:, 10] = stim_spec.stim_block.values
            val[:, 11] =  p2p[c, :]  # LL_peak_ratio[c, :]  # ratio
            # set stimulation channels to nan
            val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 3] = 1
            val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 2] = np.nan
            # if its the recovery channel, check if strange peak is appearing
            pks = np.max(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1)
            pks_loc = np.argmax(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1) + np.int64(
                (t_0 - 0.05) * Fs)
            ix = np.where(
                (pks > 500) & (pks_loc > np.int64((t_0 - 0.005) * Fs)) & (pks_loc < np.int64((t_0 + 0.008) * Fs)))
            # original stim number:
            sn = stim_spec.StimNum.values[ix]
            rec_chan = stimlist.loc[np.isin(stimlist.StimNum, sn - 1), 'ChanP'].values
            rec_chan = SM2IX(rec_chan, StimChanSM, np.array(StimChanIx))
            if np.isin(c, rec_chan):
                val[ix, 3] = 1

            data_LL = np.concatenate((data_LL, val), axis=0)

        data_LL = data_LL[1:-1, :]  # remove first row (dummy row)
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "P2P": data_LL[:, 11], "Artefact": data_LL[:, 3],
             "nLL": data_LL[:, 2], "Int": data_LL[:, 4],
             'Condition': data_LL[:, 5], 'Hour': data_LL[:, 6], "Block": data_LL[:, 10], "Sleep": data_LL[:, 9],
             "Num": data_LL[:, 7],"Num_block": data_LL[:, 7], "Date": data_LL[:, 8]})

        # distance
        for s in np.unique(LL_all.Stim):
            s = np.int64(s)
            for c in np.unique(LL_all.Chan):
                c = np.int64(c)
                LL_all.loc[(LL_all.Stim == s) & (LL_all.Chan == c), 'd'] = np.round(
                    distance.euclidean(coord_all[s], coord_all[c]), 2)
        # remove bad channels
        LL_all.loc[(LL_all.Chan).isin(bad_chans), 'Artefact'] = 1
        LL_all.loc[(LL_all.Chan).isin(bad_chans), 'Artefact'] = 1
        # file = path_patient + '/Analysis/InputOutput/' + cond_folder + '/data/con_trial'+block_l+'.csv'
        # LL_all.to_csv(file, index=False, header=True)  # scat_plot = scat_plot.fillna(method='ffill')
        # print(file + ' -- stored')
    else:
        data_LL = np.zeros((1, 12))
        data_LL[:,2:4] = np.nan
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "P2P": data_LL[:, 2], "Noise": data_LL[:, 3],
             "nLL": data_LL[:, 2], "Int": data_LL[:, 4],
             'Condition': data_LL[:, 5], 'Hour': data_LL[:, 6], "Block": data_LL[:, 10], "Sleep": data_LL[:, 9],
             "Num": data_LL[:, 7],"Num_block": data_LL[:, 7], "Date": data_LL[:, 8]})
    return LL_all

def get_LL_all_cond(EEG_resp, stimlist, lbls, bad_chans, path_patient, cond_folder='Ph', w=0.25, Fs=500):
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM,StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(stimlist,
                                                                                                          lbls)
    data_LL = np.zeros((1, 11))  # RespChan, Int, LL, LLnorm, State
    stim_spec = stimlist[(stimlist.noise == 0)]  # &(stimlist.noise ==0)
    stimNum = stim_spec.StimNum.values  # [:,0]
    #resps = EEG_resp[:, stimNum, :]
    resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
    ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))
    IPIs = np.expand_dims(np.array(stim_spec.IPI_ms.values), 1)
    LL = LLf.get_LL_both(data=resps, Fs=Fs, IPI=IPIs, t_0=1, win=w)
    LL_trial = LLf.get_LL_all(resps[:, :, int(1 * Fs):int(1.5 * Fs)], Fs, 0.25, 1, IPIs)
    LL_peak = np.max(LL_trial, 2)
    pk_start = 0.5
    for c in range(len(LL)):
        val = np.zeros((LL.shape[1], 11))
        val[:, 0] = c  # response channel
        val[:, 1] = ChanP1
        val[:, 4] = stim_spec.Int_prob.values  # Intensity
        val[:, 2] = LL[c, :, 1]  # PP
        val[:, 3] = LL_peak[c, :]  # PP
        val[:, 6] = stim_spec['h'].values
        val[:, 5] = stim_spec['condition'].values
        val[:, 7] = stimNum
        val[:, 8] = stim_spec.date.values
        val[:, 9] = stim_spec.sleep.values
        val[:, 10] = stim_spec.stim_block.values
        # set stimulation channels to nan
        val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 2] = np.nan
        # if its the recovery channel, check if strange peak is appearing

        pks = np.max(abs(resps[c, :, np.int64(pk_start * Fs):np.int64(1.5 * Fs)]), 1)
        pks_loc = np.argmax(abs(resps[c, :, np.int64(pk_start * Fs):np.int64(1.5 * Fs)]), 1) + np.int64(pk_start * Fs)
        # ix: trials where there is a strong peak during stim period in specific channel
        ix = np.where((pks > 100) & (pks_loc > np.int64(0.95 * Fs)) & (pks_loc < np.int64(1.005 * Fs)))
        # check the stimulation nuber of those trials
        sn = stim_spec.StimNum.values[ix]
        # get channels that were stimulating befor (stimnum-1) and might be recovering
        rec_chan = stimlist.loc[np.isin(stimlist.StimNum, sn - 1), 'ChanP'].values
        rec_chan = bf.SM2IX(rec_chan, StimChanSM, np.array(StimChanIx))

        ix_c = np.where(bf.check_inStimChan(c, rec_chan, labels_clinic) == 1)
        ix_real = np.intersect1d(ix, ix_c)
        val[ix_real, 2] = np.nan
        # same procedure for weir behavor before stimulation
        voltage_rec = np.percentile(abs(resps[c, :, 0:np.int64(1 * Fs)]), 90, 1)
        ix = np.where(voltage_rec > 500)
        sn = stim_spec.StimNum.values[ix]
        rec_chan = stimlist.loc[np.isin(stimlist.StimNum, sn - 1), 'ChanP'].values
        rec_chan = bf.SM2IX(rec_chan, StimChanSM, np.array(StimChanIx))
        ix_c = np.where(bf.check_inStimChan(c, rec_chan, labels_clinic) == 1)
        ix_real = np.intersect1d(ix, ix_c)
        val[ix_real, 2] = np.nan

        data_LL = np.concatenate((data_LL, val), axis=0)

    data_LL = data_LL[1:-1, :]  # remove first row (dummy row)
    LL_all = pd.DataFrame(
        {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "LL_peak": data_LL[:, 3],
         "nLL": data_LL[:, 2], "Int": data_LL[:, 4],
         'Condition': data_LL[:, 5], 'Hour': data_LL[:, 6], "Block": data_LL[:, 10], "Sleep": data_LL[:, 9],
         "Num": data_LL[:, 7], "Date": data_LL[:, 8]})

    # distance
    for s in np.unique(LL_all.Stim):
        s = np.int64(s)
        for c in np.unique(LL_all.Chan):
            c = np.int64(c)
            LL_all.loc[(LL_all.Stim == s) & (LL_all.Chan == c), 'd'] = np.round(
                distance.euclidean(coord_all[s], coord_all[c]), 2)
    # remove bad channels
    LL_all.loc[(LL_all.Chan).isin(bad_chans), 'LL'] = np.nan
    LL_all.loc[(LL_all.Chan).isin(bad_chans), 'nLL'] = np.nan
    file = path_patient + '/Analysis/InputOutput/' + cond_folder + '/data/con_trial.csv'
    LL_all.to_csv(file, index=False, header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    print(file + ' -- stored')
    return LL_all
def remove_art(EEG_resp, con_trial, Fs =500):
    # remove LL that are much higher than the mean
    if  ('zLL' in con_trial.columns):
        con_trial = con_trial.drop(columns= 'zLL')

    con_trial.insert(0, 'zLL', 0)

    con_trial.zLL = con_trial.groupby(['Stim', 'Chan', 'Int'])['LL_peak'].transform(lambda x: (x - x.mean()) / x.std()).values
    con_trial.loc[(con_trial.zLL > 4), 'LL'] = np.nan
    con_trial = con_trial.drop(columns='zLL')

    # remove trials that have artefacts (high voltage values)
    chan, trial = np.where(np.max(abs(EEG_resp), 2) > 2000)
    for i in range(len(trial)):
        con_trial.loc[(con_trial.Chan == chan[i]) & (con_trial.Num == trial[i]), 'LL'] = np.nan


    resp_BL  = abs(ff.lp_filter(EEG_resp, 2,Fs))
    resp_BL = resp_BL[:,:,0:int(Fs)]
    resp_BL[resp_BL<100] = 0
    AUC_BL = np.trapz(resp_BL, dx=1)
    chan, trial = np.where(AUC_BL>20000)
    for i in range(len(trial)):
        con_trial.loc[(con_trial.Chan == chan[i]) & (con_trial.Num == trial[i]), 'LL'] = np.nan

    # remove unrealistic high LL
    con_trial.loc[con_trial.LL > 20, 'LL'] = np.nan

    # con_trial.insert(3, 'zLL', 0)
    # con_trial.insert(3, 'nLL', 0)
    ## normalize after removing artifacts
    #con_trial.zLL = con_trial.groupby(['Stim', 'Chan'])['LL'].transform(lambda x: (x - x.mean()) / x.std()).values
    # con_trial.nLL = con_trial.groupby(['Stim', 'Chan'])['LL'].transform(lambda x: x / x.max()).values

    return con_trial

def get_LL_thr(EEG_resp, LL_all, labels_all, path_patient, n_trial=3):
    ## get threshoold value for each response channel (99th and 95h)
    chan_thr = np.zeros((len(labels_all), 4))
    # todo: only channels that are not bad to save time

    for rc in range(len(labels_all)):
        chan_thr[rc, :] = get_sig_thr(rc, LL_all, EEG_resp, n_trial)
    data_A = pd.DataFrame(chan_thr, columns=['99', '95', 'std', 'mean'])
    data_A.to_csv(path_patient + '/Analysis/InputOutput/LL/data/chan_sig_thr.csv', index=False,
                  header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    print('Data stored')
    print(path_patient + '/Analysis/InputOutput/LL/data/chan_sig_thr.csv')
    return chan_thr


def LL_mx(EEG_trial, Fs=500, w=0.25, t0=1.01):
    # calculate mean response and get LL (incl peak)
    resp = ff.lp_filter(np.mean(EEG_trial, 0), 20, Fs)
    LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp, axis=0), 0), Fs, w, 1, 0)
    LL_resp = LL_resp[0, 0]
    mx = np.max(LL_resp[np.int64((t0 + w / 2) * Fs):np.int64((t0 + w) * Fs)])
    mx_ix = np.argmax(LL_resp[np.int64((t0 + w / 2) * Fs):np.int64((t0 + w) * Fs)])
    return mx, mx_ix, LL_resp


def sig_resp(mean, thr, w=0.25, Fs=500):
    # check whether a mean response is a significant CCEP based on a pre-calculated threshold thr
    mean = ff.lp_filter(mean, 45, Fs)
    LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(mean, axis=0), 0), Fs, w, 1, 0)
    LL_resp = LL_resp[0, 0]
    mx = np.max(LL_resp[np.int64((1.01 + w / 2) * Fs):np.int64((1.01 + w) * Fs)])
    max_ix = np.argmax(LL_resp[np.int64((1.01 + w / 2) * Fs):np.int64((1.01 + w) * Fs)])
    if mx > thr:
        sig = 1
    else:
        sig = 0
    return LL_resp, mx, max_ix, sig


def get_sig_thr(rc, LL_CCEP, EEG_resp, t_num, Fs=500, fig_path='no'):
    # t_num = number of trials included for mean calculation, IO =3
    BL_times = np.concatenate(
        [np.arange(0, 0.5, 0.01), np.arange(1.6, 2, 0.01)])  # times wihtout stimulation, 0-0.5s, 1.6 - 2.5
    n = 200  # number of surrogates
    LL_surr = np.zeros((n, 1))
    list_surr = LL_CCEP[(LL_CCEP['d'] > 8) & (LL_CCEP['Chan'] == rc) & ~(LL_CCEP['Stim'] == rc) & ~np.isnan(
        LL_CCEP.LL.values)]  # take BL when rc is not stimulating and not during noise
    list_surr = list_surr[~np.isnan(list_surr.LL.values)]
    stimNum = list_surr.Num.values.astype('int')
    thr = np.zeros(4, )
    if len(stimNum) > 0:
        for k in range(n):
            t0 = np.random.choice(np.round(BL_times, 2))
            stimNum_choice = np.random.choice(stimNum, t_num)
            EEG_trial = EEG_resp[rc, stimNum_choice,
                        np.int64((t0) * Fs):np.int64((t0 + 0.4) * Fs)]  # np.flip(EEG_resp[rc,stimNum,:],1)
            LL_surr[k, 0], _, _ = LL_mx(EEG_trial, t0=0)

        thr[0] = np.percentile(LL_surr[:, 0], 99)
        thr[1] = np.percentile(LL_surr[:, 0], 95)
        thr[2] = np.nanstd(LL_surr[:, 0])
        thr[3] = np.nanmean(LL_surr[:, 0])
        if fig_path != 'no':
            fig = plt.figure(figsize=(5, 5))
            plt.title('surrogates - ' + labels_all[rc])
            plt.hist(LL_surr[:, 0])
            plt.axvline(thr[0], c=[1, 0, 0], label='99%')
            plt.axvline(thr[1], c=[1, 0, 0], label='90%')
            plt.axvline(np.mean(LL_surr[:, 0]) + np.std(LL_surr[:, 0]), c=[0, 0, 0], label='mean +std')
            plt.xlabel('LL [250ms]')
            plt.xlim([0, np.max([2, 1.1 * max(LL_surr[:, 0])])])
            plt.legend()
            plt.savefig(fig_path)
            plt.close(fig)  # close the figure window
    return thr

def get_peaks_all(LL_CCEP, EEG_resp,M_N1peaks, t0=1, Fs=500):
    new_lab = ['N1', 'N2', 'sN1', 'sN2', 't_N1', 't_N2']
    for l in new_lab:
        if l not in LL_CCEP:
            LL_CCEP.insert(6, l, np.nan)
    data = LL_CCEP[~(np.isnan(LL_CCEP.LL.values))]  # [~(np.isnan(LL_CCEP.LL.values))]
    s=0
    for sc in np.unique(data.Stim).astype('int'):
        if len(M_N1peaks)>len(np.unique(data.Stim)):
            sc_ix = sc
        else:
            sc_ix = s
        for rc in np.unique(data.loc[data.Stim == sc, 'Chan']).astype('int'):
            lists = LL_CCEP[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
            stimNum_all = lists.Num.values.astype('int')
            if np.mean(abs(M_N1peaks[sc_ix,rc,:]))>0:
                resps_trials = EEG_resp[rc, stimNum_all, :]
                peaks = Pkf.get_peaks_trials(resps_trials, M_N1peaks[sc_ix,rc,:], t_0=1, Fs=500)
            # slope = LLf.pk_lin_fit(resps_trials, peaks, fig=0)
            else:
                peaks = np.zeros((len(stimNum_all),4,2))
                peaks[:,:,:] =np.nan
            LL_CCEP.loc[
                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc), 'N1'] = abs(
                peaks[:,1, 0] - peaks[:,0, 0])
            LL_CCEP.loc[
                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) , 'N2'] = abs(
                peaks[:,1, 0] - peaks[:,2, 0])
            LL_CCEP.loc[
                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc), 't_N1'] = (peaks[:,0, 1]/Fs)-t0
            LL_CCEP.loc[
                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) , 't_N2'] = (peaks[:,2,1]/Fs)-t0
            LL_CCEP.loc[(LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc), 'sN1'] = (peaks[:, 1, 0] - peaks[:, 0, 0]) / (peaks[:, 1, 1] - peaks[:, 0, 1]) / Fs * 1000
            LL_CCEP.loc[(LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc), 'sN2'] = (peaks[:, 2, 0] - peaks[:, 1, 0]) / (
                        peaks[:, 2, 1] - peaks[:, 1, 1]) / Fs * 1000 # uV/ms

            # LL_CCEP.loc[
            #     (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) & (LL_CCEP.Num == StimNum[i]), ['sN1','sN2',]] = slope[0:2]
        s=s+1
    return LL_CCEP

def concat_resp_condition(subj, cond_folder='CR'):
    folder = 'InputOutput'
    path_patient_analysis = 'T:\EL_experiment\Projects\EL_experiment\Analysis\Patients\\' + subj
    files = glob(path_patient_analysis + '\\' + folder + '\\data\\Stim_list_*'+cond_folder+'*')
    files = np.sort(files)
    # prots           = np.int64(np.arange(1, len(files) + 1))  # 43
    stimlist = []
    EEG_resp = []
    conds = np.empty((len(files),), dtype=object)
    for p in range(len(files)):
        file = files[p]
        #file = glob(self.path_patient + '/Analysis/'+folder+'/data/Stim_list_' + str(p) + '_*')[0]
        idxs       = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
        cond       = ntpath.basename(file)[idxs[-2] - 2:idxs[-2] ]#ntpath.basename(file)[idxs[-2] + 2:-4]  #
        conds[p]   = cond
        EEG_block  = np.load(path_patient_analysis + '\\' + folder + '\\data\\All_resps_' + file[-11:-4]+ '.npy')
        print(str(p+1)+'/'+str(len(files))+' -- All_resps_' + file[-11:-4])
        stim_table = pd.read_csv(file)
        stim_table['type'] = cond
        if len(stimlist) == 0:
            EEG_resp = EEG_block
            stimlist = stim_table
        else:
            EEG_resp = np.concatenate([EEG_resp, EEG_block], axis=1)
            stimlist = pd.concat([stimlist, stim_table])
    stimlist  = stimlist.drop(columns="StimNum", errors='ignore')
    stimlist        = stimlist.fillna(0)
    stimlist = stimlist.reset_index(drop=True)
    col_drop = ["StimNum",'StimNum.1', 's', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
    for d in range(len(col_drop)):
        if (col_drop[d] in stimlist.columns):
            stimlist = stimlist.drop(columns=col_drop[d])
    stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)
    np.save(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_'+cond_folder+'.npy', EEG_resp)
    stimlist.to_csv(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\stimlist_' + cond_folder + '.csv', index=False,
                    header=True)  # scat_plot
    print('data stored')
    return EEG_resp, stimlist



def get_peaks(LL_CCEP, EEG_resp, t_0=1, Fs=500):
    new_lab = ['N1', 'N2', 'P2', 'sN1', 'sN2', 'sP2', 't_N1', 't_N2', 't_P1']
    for l in new_lab:
        if l not in LL_CCEP:
            LL_CCEP.insert(6, l, np.nan)
    data = LL_CCEP[~(np.isnan(LL_CCEP.LL.values))]  # [~(np.isnan(LL_CCEP.LL.values))]

    for sc in np.unique(data.Stim).astype('int'):
        for rc in np.unique(data.loc[data.Stim == sc, 'Chan']).astype('int'):
            StimNum = data.loc[(data.Stim == sc) & (data.Chan == rc) & (data.Int > 8), 'Num'].values.astype('int')
            if len(StimNum) > 2:
                resp_all = bf.zscore_CCEP(np.mean(ff.lp_filter(EEG_resp[rc, StimNum, :], 40, Fs), 0))
                if np.max(abs(resp_all)) > 5:
                    ## get start resp
                    peaks_all, properties_all = scipy.signal.find_peaks(
                        abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]), height=1.5, prominence=0.05,
                        distance=0.03 * Fs, width=1)  #

                    if len(peaks_all) > 0:
                        peaks_all = peaks_all[0]
                        w = scipy.signal.peak_widths(abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]), [peaks_all],
                                                     rel_height=0.5)[0]

                        start_resp = (peaks_all - w) / Fs - 0.01

                        if start_resp < 0.01:
                            start_resp = 0

                    pk, peak_s, p = LLf.get_peaks_all(resp_all, start_resp)
                    # pk_all          = resp_all[pk]
                    StimNum = data.loc[(data.Stim == sc) & (data.Chan == rc), 'Num'].values.astype(
                        'int')
                    if np.mean(pk) > 500:
                        # for Int in np.unique(data.loc[(data.Stim==sc)&(data.Chan==rc), 'Int'].values):
                        for i in range(len(StimNum)):
                            resp_raw = ff.lp_filter(EEG_resp[rc, StimNum[i], :], 40, Fs)
                            peaks = LLf.get_peaks_trial(resp_raw, peak_s, p=p, t_0=1, Fs=500)
                            slope = LLf.pk_lin_fit(resp_raw, peaks, fig=0, t_0=1, Fs=500)

                            LL_CCEP.loc[
                                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) & (LL_CCEP.Num == StimNum[i]), 'N1'] = abs(
                                peaks[1, 1] - peaks[0, 1])
                            LL_CCEP.loc[
                                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) & (LL_CCEP.Num == StimNum[i]), 'N2'] = abs(
                                peaks[1, 1] - peaks[2, 1])
                            LL_CCEP.loc[
                                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) & (LL_CCEP.Num == StimNum[i]), 'P2'] = abs(
                                peaks[3, 1] - peaks[2, 1])
                            LL_CCEP.loc[
                                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) & (LL_CCEP.Num == StimNum[i]), 't_N1'] = \
                            peaks[0, 0]
                            LL_CCEP.loc[
                                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) & (LL_CCEP.Num == StimNum[i]), 't_N2'] = \
                            peaks[2, 0]
                            LL_CCEP.loc[
                                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) & (LL_CCEP.Num == StimNum[i]), 't_P1'] = \
                            peaks[1, 0]
                            LL_CCEP.loc[
                                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) & (LL_CCEP.Num == StimNum[i]), ['sN1',
                                                                                                            'sN2',
                                                                                                            'sP2']] = slope
                            # LL_CCEP.loc[(LL_CCEP.Stim == sc)&(LL_CCEP.Chan == rc)&(LL_CCEP.Num == StimNum[i]), 'sN2'] = abs(peaks[1,1]-peaks[2,1])
                            # LL_CCEP.loc[(LL_CCEP.Stim == sc)&(LL_CCEP.Chan == rc)&(LL_CCEP.Num == StimNum[i]), 'sP2'] = abs(peaks[3,1]-peaks[2,1])
                        # for l in ['N1', 'N2']:
                        #     m = np.mean(LL_CCEP.loc[(LL_CCEP.Condition == 1)&(LL_CCEP.Stim == sc)&(LL_CCEP.Chan == rc), l])
                        #     LL_CCEP.loc[(LL_CCEP.Stim == sc)&(LL_CCEP.Chan == rc), 'n'+l] = LL_CCEP.loc[(LL_CCEP.Stim == sc)&(LL_CCEP.Chan == rc), l]/m
    return LL_CCEP


def get_peaks_mean(con_trial, con_mean, EEG_resp, Condition, t_0=1, Fs=500):
    new_lab = ['N1', 'N2', 'P2', 'sN1', 'sN2', 'sP2', 't_N1', 't_N2', 't_P1']
    for l in new_lab:
        if l not in con_mean:
            con_mean.insert(6, l, np.nan)
    data = con_trial[~(np.isnan(con_trial.LL.values))]

    for sc in np.unique(data.Stim).astype('int'):
        for rc in np.unique(data.loc[data.Stim == sc, 'Chan']).astype('int'):
            StimNum = data.loc[(data.Stim == sc) & (data.Chan == rc) & (data.Int > 6), 'Num'].values.astype('int')
            if len(StimNum) > 4:
                resp_all = bf.zscore_CCEP(np.mean(ff.lp_filter(EEG_resp[rc, StimNum, :], 40, Fs), 0))
                if np.max(abs(resp_all)) > 5:
                    ## get start resp
                    peaks_all, properties_all = scipy.signal.find_peaks(
                        abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]), height=1.5, prominence=0.05,
                        distance=0.03 * Fs, width=1)  #

                    if len(peaks_all) > 0:
                        peaks_all = peaks_all[0]
                        w = scipy.signal.peak_widths(abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]), [peaks_all],
                                                     rel_height=0.5)[0]

                        start_resp = (peaks_all - w) / Fs - 0.01

                        if start_resp < 0.01:
                            start_resp = 0

                    pk, peak_s, p = LLf.get_peaks_all(resp_all, start_resp)
                    # pk_all          = resp_all[pk]
                    for day in np.unique(data.Date):
                        for cond in np.unique(data[Condition]):
                            for Int in np.unique(data.Int):
                                StimNum = data.loc[
                                    (data.Date == day) & (data[Condition] == cond) & (data.Stim == sc) & (
                                                data.Chan == rc) & (
                                            data.Int == Int), 'Num'].values.astype('int')
                                resp_raw = np.mean(ff.lp_filter(EEG_resp[rc, StimNum, :], 45, Fs), 0)
                                peaks = LLf.get_peaks_trial(resp_raw, peak_s, p=p, t_0=1, Fs=500)
                                slope = LLf.pk_lin_fit(resp_raw, peaks, fig=0, t_0=1, Fs=500)

                                con_mean.loc[
                                    (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc) & (
                                            con_mean.Int == Int), 'N1'] = abs(peaks[1, 1] - peaks[0, 1])
                                con_mean.loc[
                                    (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc) & (
                                            con_mean.Int == Int), 'N2'] = abs(peaks[1, 1] - peaks[2, 1])
                                con_mean.loc[
                                    (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc) & (
                                            con_mean.Int == Int), 'P2'] = abs(peaks[3, 1] - peaks[2, 1])
                                con_mean.loc[
                                    (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc) & (
                                            con_mean.Int == Int), 't_N1'] = peaks[0, 0]
                                con_mean.loc[
                                    (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc) & (
                                            con_mean.Int == Int), 't_N2'] = peaks[2, 0]
                                con_mean.loc[
                                    (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc) & (
                                            con_mean.Int == Int), 't_P1'] = peaks[1, 0]
                                con_mean.loc[
                                    (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc) & (
                                            con_mean.Int == Int), ['sN1', 'sN2', 'sP2']] = slope

    return con_mean


def get_LL_mean(EEG_resp, LL_all, chan_thr, labels_all, labels_region, cond_sel, path_patient):
    cond_folder = 'Ph'  # 'Ph', 'Sleep', 'CR'
    Condition = 'Condition'
    if cond_sel == 'Hour':
        Condition = 'Hour'
        cond_folder = 'CR'
        cond_sel = 'Block'

    # calculates for each channel pair and intensity the mean CCEP and LL and checks whether it is significant
    LL_mean = np.zeros((1, 10))
    lists = LL_all[(LL_all['Int'] > 0)]
    lists = lists[~np.isnan(lists.LL.values)]
    stims = np.unique(lists.Stim)
    Int_selc = np.unique(lists.Int)
    # todo: change to block instead of hour
    # cond_val = np.unique(lists[cond_sel])
    for sc in stims:
        sc = np.int64(sc)
        resps = np.unique(lists.loc[(lists.Stim == sc), 'Chan']).astype('int')
        for rc in resps:
            for day in np.unique(lists.Date):
                cond_val = np.unique(lists.loc[lists.Date == day, cond_sel])  # block or Ph
                # for block in np.unique(lists.Block):  # each condition
                for j in range(len(cond_val)):  # each block or condition in specific day
                    for i in range(len(Int_selc)):  # each intensity
                        dati = lists[(lists.Date == day) & (lists.Int == Int_selc[i]) & (lists.Stim == sc) & (
                                    lists.Chan == rc) & (
                                             lists[cond_sel] == cond_val[j])]
                        if len(dati) > 0:
                            resp = np.nanmean(EEG_resp[rc, dati.Num.values.astype('int'), :], 0)
                            LL_resp, mx, mx_ix, sig = sig_resp(resp, chan_thr[rc, 0])

                            val = np.zeros((1, 10))
                            val[0, 0] = rc  # response channel
                            val[0, 1] = sc  # response channel
                            val[0, 2] = mx
                            val[0, 3] = Int_selc[i]
                            val[0, 4] = np.nanmean(dati.d)
                            val[0, 5] = sig
                            val[0, 6] = cond_val[j]  # block or Ph_condition
                            # todo: change to block and add hour with bincount
                            val[0, 7] = day
                            val[0, 8] = np.bincount(dati.Hour).argmax()  # stim_spec.sleep.values
                            # val[:, 8] = stim_spec.sleep.values
                            val[0, 9] = np.bincount(dati.Sleep).argmax()

                            LL_mean = np.concatenate((LL_mean, val), axis=0)
    LL_mean = LL_mean[1:, :]  # remove first row (dummy row)
    data_A = pd.DataFrame(
        {"Chan": LL_mean[:, 0], "Stim": LL_mean[:, 1], "LL": LL_mean[:, 2], "Sig": LL_mean[:, 5], "Int": LL_mean[:, 3],
         "d": LL_mean[:, 4], cond_sel: LL_mean[:, 6], 'Hour': LL_mean[:, 8], 'Sleep': LL_mean[:, 9],
         'Date': LL_mean[:, 7]})  # , "Sig_Con": LL_mean[:, 7]
    # normalize by max LL
    data_A.insert(3, 'nLL', 0)
    # data_A.zLL = data_A.groupby(['Stim', 'Chan'])['LL'].transform(lambda x: (x - x.mean()) / x.std()).values
    data_A.nLL = data_A.groupby(['Stim', 'Chan'])['LL'].transform(lambda x: x / x.max()).values

    file = path_patient + '/Analysis/InputOutput/' + cond_folder + '/data/LL_sig_intensity_' + Condition + '.csv'
    data_A.to_csv(file, index=False, header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    print('data stored --  ' + file)
    return data_A


def get_IO_summary(LL_mean, cond_sel, labels_all, labels_region, path_patient):
    cond1 = 'Condition'  # 'condition', 'h'
    cond_folder = 'Ph'  # 'Ph', 'Sleep', 'CR'
    Condition = 'Condition'
    if cond_sel == 'Hour':
        Condition = 'Hour'
        cond_folder = 'CR'
    data_mean = np.zeros((1, 9))
    data_test = LL_mean[LL_mean.LL > 0]  # no artefacts
    stims = np.unique(data_test.Stim)
    Int_all = np.unique(data_test.Int)
    for sc in stims:  # repeat for each stimulation channel
        sc = np.int64(sc)
        resps = np.unique(data_test.loc[(data_test.Stim == sc), 'Chan'])
        for rc in resps:
            rc = np.int64(rc)
            LL0 = np.min((data_test.loc[(data_test.Stim == sc) & (data_test.Chan == rc), 'LL norm']).values)
            cond_val = np.unique(data_test[cond_sel])

            for day in np.unique(data_test.Date):
                for j in range(len(cond_val)):
                    dati = data_test[(data_test.Date == day) &
                                     (data_test.Stim == sc) & (data_test.Chan == rc) & (
                                                 data_test[Condition] == cond_val[j])]
                    if len(dati) > 3:
                        val = np.zeros((1, 9))
                        val[0, 0] = rc  # response channel
                        val[0, 1] = sc
                        val[0, 6] = cond_val[j]  # condition
                        val[0, 4] = np.nanmean(dati.d)  # distance
                        val[0, 8] = day  # date
                        val[0, 2] = trapz(dati['LL norm'].values - LL0, dati['Int'].values) / np.max(Int_all)  # AUC
                        Int_min = np.unique(dati.loc[dati.Sig == 1, 'Int'])  # all ints inducing CCEP
                        if len(np.unique(dati.loc[dati.Sig == 0, 'Int'])) > 0:  # if not all int inducing CCEPs
                            # only int with higher int also inducing CCEP
                            Int_min = np.unique(dati.loc[dati.Sig == 1, 'Int'])[
                                np.where(Int_min - np.unique(dati.loc[dati.Sig == 0, 'Int'])[-1] > 0)]
                        if (len(Int_min) > 0) and (np.mean(dati.loc[dati.Int > Int_all[-3], 'Sig']) == 1):

                            Int_min = Int_min[0]
                            val[0, 3] = Int_min
                            val[0, 5] = dati.loc[dati.Int == Int_min, 'LL norm'].values
                            val[0, 7] = 1
                            data_mean = np.concatenate((data_mean, val), axis=0)
                        else:
                            Int_min = 0
                            val[0, 3] = np.nan
                            val[0, 5] = 1
                            val[0, 7] = 0

    data_mean = data_mean[1:-1, :]  # remove first row (dummy row)
    IO_mean = pd.DataFrame(
        {"RespC": data_mean[:, 0], "StimC": data_mean[:, 1], "AUC": data_mean[:, 2], "MPI": data_mean[:, 3],
         "d": data_mean[:, 4], Condition: data_mean[:, 6], "Date": data_mean[:, 8], "Sig": data_mean[:, 7]})

    for c in range(len(labels_all)):
        IO_mean.loc[(IO_mean.RespC == c), "RespR"] = labels_region[c]
        IO_mean.loc[(IO_mean.RespC == c), "Resp"] = labels_all[c]
        IO_mean.loc[(IO_mean.StimC == c), "StimR"] = labels_region[c]
        IO_mean.loc[(IO_mean.StimC == c), "Stim"] = labels_all[c]

    IO_mean = IO_mean.drop(IO_mean[IO_mean['RespR'] == 'OUT'].index)
    # IO_mean=IO_mean.drop(IO_mean[IO_mean['RespR']=='WM'].index)
    file = path_patient + '/Analysis/InputOutput/' + cond_folder + '/data/IO_mean_' + Condition + '.csv'
    IO_mean.to_csv(file, index=False,
                   header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    print('data stored -- ' + path_patient + '/Analysis/InputOutput/LL/data/IO_mean_' + Condition + '.csv')
    return IO_mean


def plot_raw_LL_IO_cond(sc, rc, LL_all, LL_mean, EEG_resp, chan_thr, labels_all, path_patient):
    dat = LL_all[(LL_all['Stim'] == sc) & (LL_all['Chan'] == rc)]
    conds = [1, 3]  # np.unique(dat.Condition)
    w = 0.25
    fig, axs = plt.subplots(len(conds), 3, figsize=(15, 8), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    plt.close(fig)  # todo: find better solution
    fig = plt.figure(figsize=(15, 8), facecolor='w', edgecolor='k')
    #
    gs = fig.add_gridspec(len(conds), 3, width_ratios=[1, 1, 2])  # GridSpec(4,1, height_ratios=[1,2,1,2])
    for i in range(len(conds)):
        axs[2 * i + 0] = fig.add_subplot(gs[i, 0])
        axs[2 * i + 1] = fig.add_subplot(gs[i, 1])

    axIO = fig.add_subplot(gs[:, 2])
    plt.suptitle(labels_all[np.int64(sc)] + ', Resp: ' + labels_all[np.int64(rc)] + ', d=' + str(
        np.round(np.mean(dat.d), 1)) + 'mm', y=0.95)
    limy_LL = 3  # limits for LL plot
    limy_CCEP = 200
    Int_selc = np.unique(dat.Int)
    colors_Int = np.zeros((len(Int_selc), 3))
    colors_Int[:, 0] = np.linspace(0, 1, len(Int_selc))
    LL0 = np.min((LL_mean.loc[(LL_mean.Stim == sc) & (LL_mean.Chan == rc), 'LL norm']).values)
    mx_LL = 1
    for j in range(len(conds)):
        con_sel = np.int64(conds[j])
        Int_selc = np.unique(dat.loc[(dat.Stim == sc) & (dat.Chan == rc) & (dat.Condition == con_sel), 'Int'])
        for i in range(len(Int_selc)):
            dati = dat[(dat.Int == Int_selc[i]) & (dat.Stim == sc) & (dat.Chan == rc) & (dat.Condition == con_sel)]
            if len(dati) > 0:
                resp = np.nanmean(EEG_resp[rc, dati.Num.values.astype('int'), :], 0)
                LL_resp, mx, max_ix, sig = sig_resp(resp, chan_thr[rc, 1])
                mx_norm = LL_mean.loc[(LL_mean.Stim == sc) & (LL_mean.Chan == rc) & (LL_mean.Int == Int_selc[i]) & (
                        LL_mean.Condition == con_sel), 'LL norm']
                sig = np.mean(LL_mean.loc[(LL_mean.Stim == sc) & (LL_mean.Chan == rc) & (LL_mean.Int == Int_selc[i]) & (
                        LL_mean.Condition == con_sel), 'Sig'])
                axs[0 + 2 * j].plot(x_ax, ff.lp_filter(resp, 40, Fs), c=colors_Int[i], alpha=0.5 + 0.5 * sig,
                                    linewidth=1.2 * (0.5 + sig))
                axs[0 + 2 * j].set_xlim(-0.2, 0.5)
                axs[1 + 2 * j].plot(x_ax, LL_resp, c=colors_Int[i], alpha=0.5 + 0.5 * sig, linewidth=1.2 * (0.5 + sig))
                axs[1 + 2 * j].plot(0.01 + w / 2 + max_ix / Fs, mx, marker='+', c=[0, 0, 0], alpha=0.7 + 0.3 * sig,
                                    markersize=10)
                axs[1 + 2 * j].set_xlim(-0.2, 0.5)
                # axIO.plot(Int_selc[i], mx, marker='o', markersize=10, c = cond_colors[con_sel], alpha=0.2+0.8*sig)

                axIO.plot(Int_selc[i], mx_norm, marker='o', markersize=10, c=cond_colors[con_sel],
                          alpha=0.2 + 0.8 * sig)
                limy_LL = np.nanmax([limy_LL, mx])
                limy_CCEP = np.nanmax([limy_CCEP, np.max(abs(resp))])
                mx_LL = np.max([mx_LL, mx_norm.values[0]])
        # y = (data_A.loc[(data_A.Stim==sc)&(data_A.Chan==rc)&(data_A.Condition==con_sel), 'LL norm']*data_A.loc[(data_A.Stim==sc)&(data_A.Chan==rc)&(data_A.Condition==con_sel), 'Sig']).values
        y = (LL_mean.loc[
            (LL_mean.Stim == sc) & (LL_mean.Chan == rc) & (LL_mean.Condition == con_sel), 'LL norm']).values - LL0

        axIO.plot(Int_selc[i], mx_norm, marker='o', markersize=10, c=cond_colors[con_sel], alpha=0.2 + 0.8 * sig,
                  label=cond_labels[con_sel] + ', AUC (%): ' + str(np.round(trapz(y, Int_selc) / np.max(Int_selc), 2)))

    for i in range(len(conds) * 2):
        axs[i].axvline(0, c=[0, 0, 0])

    axIO.legend(loc='lower right')
    axIO.set_title('IO curve')
    axIO.set_ylabel('LL uv/ms [250ms] normalized')
    axIO.set_xlabel('Intensity [mA]')
    axIO.set_ylim([0, 1.1 * mx_LL])
    axIO.axhline(LL0, color="black", linestyle="--")

    axs[0].set_title('mean CCEP')
    axs[1].set_title('LL [' + str(w) + 's] of mean CCEP')
    for i in range(len(conds)):
        axs[2 * i].set_ylabel(cond_labels[np.int64(conds[i])])
        axs[2 * i].set_ylim(-limy_CCEP, limy_CCEP)
        # axs[2*i].axvspan(0.01, 0.01+w, alpha=0.05, color='blue')
    for i in range(len(conds)):
        # axs[2*i+1].axvline(0.01+w/2, c=[0,0,1], alpha = 0.5)
        axs[2 * i + 1].set_ylim(0, 1.2 * limy_LL)
    axs[2].set_xlabel('time [s]')
    axs[3].set_xlabel('time [s]')

    plt.savefig(path_patient + '/Analysis/InputOutput/LL/figures/IO_' + labels_all[sc] + '_' + labels_all[rc] + '.jpg')
    plt.savefig(path_patient + '/Analysis/InputOutput/LL/figures/IO_' + labels_all[sc] + '_' + labels_all[rc] + '.svg')
    plt.show()
