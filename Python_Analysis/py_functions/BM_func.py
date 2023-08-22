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
import tqdm
import platform
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from scipy.spatial import distance
import itertools
import math
import LL_funcs as LLf
import basic_func as bf
import Peaks_funcs as Pkf

if platform.system() == 'Windows':
    regions = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\elab_labels.xlsx", sheet_name='regions', header=0)

else:  # 'Darwin' for MAC
    regions = pd.read_excel("/Volumes/EvM_T7/PhD/EL_experiment/Patients/all/elab_labels.xlsx", sheet_name='regions',
                            header=0)

color_regions = regions.color.values
regions_G = regions.subregion.values
regions = regions.label.values

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]


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
    # SM: selected array of stim channel in SM number
    # StimChanNums: all number of stim channels in SM
    # StimChanIx: all stim channels in all channles environment
    ChanIx = np.zeros_like(SM)
    for i in range(len(SM)):
        ChanIx[i] = StimChanIx[np.where(StimChanNums == SM[i])]
    return ChanIx

def init_stimlist_columns(stimlist, StimChanSM):
    """Initialize required columns if they are not present."""
    column_defaults = {"Num_block": stimlist.StimNum, "condition": 0, "sleep": 0}
    for col, default_val in column_defaults.items():
        if col not in stimlist.columns:
            stimlist[col] = default_val
    # Filter stimlist based on conditions
    stim_spec = stimlist[(stimlist.IPI_ms == 0)&(np.isin(stimlist.ChanP, StimChanSM)) & (stimlist.noise == 0)]
    stim_spec.reset_index(drop=True, inplace=True)

    return stimlist, stim_spec

def calculate_artefact(resps, stimlist, stim_spec, t_0, Fs, c, ChanP1, StimChanSM, StimChanIx, labels_clinic):
    """Detect artefact if recording channel has high LL and was stimulating the trial before (still recovering)"""
    # pks = np.max(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1)
    # pks_loc = np.argmax(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1) + np.int64(
    #     (t_0 - 0.05) * Fs)
    #
    # ix = np.where((pks > 500) & (pks_loc > np.int64((t_0 - 0.005) * Fs)) & (pks_loc < np.int64((t_0 + 0.008) * Fs)))
    # sn = stim_spec.StimNum.values[ix]
    # rec_chan = stimlist.loc[np.isin(stimlist.StimNum, sn - 1), 'ChanP'].values
    #
    # if len(rec_chan) > 0:
    #     rec_chan = SM2IX(rec_chan, StimChanSM, np.array(StimChanIx))
    #     if np.isin(c, rec_chan):
    #         return 1 # recording channel has artefact because it's a recovering channel (stimulating just before)
    return check_inStimChan(c, ChanP1, labels_clinic) # stim channel is recording channel

def LL_BM_connection(EEG_resp, stimlist, bad_chans, coord_all, labels_clinic, StimChanSM, StimChanIx):
    Fs = 500
    w_LL = 0.25
    t_0 = 1  # time of stimulation in data

    # Init required columns in stimlist
    stimlist, stim_spec = init_stimlist_columns(stimlist,StimChanSM)

    # Analyze each channel
    data_CCEP = []
    stimNum = stim_spec.StimNum.values  # [:,0]
    noise_val = stim_spec.noise.values  # [:,0]
    stimNum_block = stim_spec.Num_block.values  # [:,0]
    resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
    ChanP1 = SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))

    ## ge tLL for each stimulation and channel
    LL_trial = LLf.get_LL_all(resps[:, :, int(t_0 * Fs):int((t_0 + 0.5) * Fs)], Fs, w_LL)
    LL_peak = np.max(LL_trial, 2)
    t_peak = np.argmax(LL_trial, 2) + int((t_0 - w_LL / 2) * Fs)
    t_peak[t_peak < (t_0 * Fs)] = t_0 * Fs
    inds = np.repeat(np.expand_dims(t_peak, 2), int(w_LL * Fs), 2)
    inds = inds + np.arange(int(w_LL * Fs))
    pN = np.min(np.take_along_axis(resps, inds, axis=2), 2)
    pP = np.max(np.take_along_axis(resps, inds, axis=2), 2)
    p2p = abs(pP - pN)
    for c in range(LL_peak.shape[0]):
        val = [
            [c, ChanP1[i], noise_val[i], stimNum_block[i], stim_spec.condition.values[i], stim_spec.date.values[i],
             stim_spec.sleep.values[i], stim_spec.stim_block.values[i], LL_peak[c, i], stim_spec.h.values[i],
             stimNum[i], p2p[c, i]]
            for i in range(LL_peak.shape[1])
        ]
        val = np.array(val)
        # # Apply artefact logic
        # for v in val:
        #     v[2] = calculate_artefact(resps, stimlist, stim_spec, t_0, Fs, c, ChanP1, StimChanSM, StimChanIx,
        #                               labels_clinic)
        chan_stimulating = check_inStimChan(c, ChanP1, labels_clinic)
        if len(chan_stimulating) > 0:
            indices = np.where(chan_stimulating == 1)[0]
            val[indices, 2] = 1

        # Convert the numpy array back to a list
        val = val.tolist()
        data_CCEP.extend(val)

    # Convert to DataFrame
    LL_CCEP = pd.DataFrame(data_CCEP, columns=["Chan", "Stim", "Artefact", "Num_block", "Condition", "Date", "Sleep",
                                               "Block", "LL", "Hour", "Num", "P2P"])

    # Mark bad channels as artefacts
    LL_CCEP.loc[LL_CCEP['Chan'].isin(bad_chans), 'Artefact'] = 1

    # distance
    for s in np.unique(LL_CCEP.Stim):
        s = np.int64(s)
        for c in np.unique(LL_CCEP.Chan):
            c = np.int64(c)
            LL_CCEP.loc[(LL_CCEP.Stim == s) & (LL_CCEP.Chan == c), 'd'] = np.round(
                distance.euclidean(coord_all[s], coord_all[c]), 2)

    return LL_CCEP  # , trial_sig

def get_peaks_all(LL_CCEP, EEG_resp,M_N1peaks, t0=1, Fs=500):
    new_lab = ['N1', 'N2', 'sN1', 'sN2', 't_N1', 't_N2']
    for l in new_lab:
        if l not in LL_CCEP:
            LL_CCEP.insert(6, l, np.nan)
    data = LL_CCEP[~(np.isnan(LL_CCEP.LL.values))]  # [~(np.isnan(LL_CCEP.LL.values))]
    for sc in np.unique(data.Stim).astype('int'):
        for rc in np.unique(data.loc[data.Stim == sc, 'Chan']).astype('int'):
            lists = LL_CCEP[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
            stimNum_all = lists.Num.values.astype('int')
            if np.mean(abs(M_N1peaks[sc,rc,:2]))>0:
                resps_trials = EEG_resp[rc, stimNum_all, :]
                peaks = Pkf.get_peaks_trials(resps_trials, M_N1peaks[sc,rc,:], t_0=1, Fs=500)
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

    return LL_CCEP
def get_SigCon_BM_trial_block(LL_CCEP, EEG_resp, labels_all):
    # cond_sel: either Condition or Hour
    Fs = 500
    t_0 = 1
    M_resp = np.zeros((len(labels_all), len(labels_all), 3)) - 1
    # (LL_CCEP[cond_sel]==cond_val)
    resp_mean = np.zeros((1, 7))
    for rc in tqdm.tqdm(range(len(labels_all))):  # for each response channel
        for sc in range(len(labels_all)):  # for each stim channel
            lists = LL_CCEP[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc) & (~np.isnan(LL_CCEP.LL.values))]
            d = np.mean(
                LL_CCEP.loc[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc), 'd'])
            lists = lists[~np.isnan(lists.LL.values)]
            stimNum_all = lists.Num_block.values.astype('int')
            val = np.zeros((1, 7))
            val[0, 0:3] = [rc, sc, d]
            if len(stimNum_all) > 0:

                resp_z = bf.zscore_CCEP(ff.lp_filter(np.mean(EEG_resp[rc, stimNum_all, :], 0), 45, Fs))
                mx = np.max(abs(resp_z[int(1.01 * Fs):int(1.4 * Fs)]))

                LL_CCEP.loc[
                    (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'Sig_block'] = mx
            else:
                LL_CCEP.loc[(LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'Sig_block'] = -1

    return LL_CCEP


def get_peaks(LL_CCEP, EEG_resp, t_0=1, Fs=500):
    new_lab = ['N1', 'N2', 'P2', 'sN1', 'sN2', 'sP2', 't_N1', 't_N2', 't_P1']
    for l in new_lab:
        if l not in LL_CCEP:
            LL_CCEP.insert(LL_CCEP.shape[1], l, np.nan)
    data = LL_CCEP[(LL_CCEP.RespC == 1) & ~(np.isnan(LL_CCEP.LL.values))]

    for sc in np.unique(data.Stim).astype('int'):
        for rc in np.unique(data.loc[data.Stim == sc, 'Chan']).astype('int'):
            StimNum = LL_CCEP.loc[
                (LL_CCEP.Stim == sc) & (LL_CCEP.Chan == rc) & ~(
                    np.isnan(LL_CCEP.LL.values)), 'Num'].values.astype(
                'int')
            if len(StimNum) > 2:
                resp_all = bf.zscore_CCEP(np.mean(ff.lp_filter(EEG_resp[rc, StimNum, :], 40, Fs), 0))
                if np.max(abs(resp_all[Fs:int(1.5 * Fs)])) > 5:
                    ## get start resp
                    peaks_all, properties_all = scipy.signal.find_peaks(
                        abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]), height=1.5, prominence=0.05,
                        distance=0.03 * Fs, width=1)  #
                    start_resp = 0
                    if len(peaks_all) > 0:
                        peaks_all = peaks_all[0]
                        w = scipy.signal.peak_widths(abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]), [peaks_all],
                                                     rel_height=0.5)[0]

                        start_resp = (peaks_all - w) / Fs - 0.01

                        if start_resp < 0.01:
                            start_resp = 0

                    pk, peak_s, p = LLf.get_peaks_all(resp_all, start_resp)
                    # pk_all          = resp_all[pk]
                    if (np.mean(pk) > Fs) & (np.max(abs(resp_all[pk[0:3]])) > 5):
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


def get_peaks_mean(con_trial, con_mean, EEG_resp, Condition, Fs=500):
    new_lab = ['N1', 'N2', 'P2', 'sN1', 'sN2', 'sP2']
    for l in new_lab:
        if l not in con_mean:
            con_mean.insert(6, l, np.nan)
    data = con_trial[(con_trial.RespC == 1) & ~(np.isnan(con_trial.zLL.values))]

    for sc in np.unique(data.Stim).astype('int'):
        for rc in np.unique(data.loc[data.Stim == sc, 'Chan']).astype('int'):
            StimNum = data.loc[(data.Stim == sc) & (data.Chan == rc), 'Num'].values.astype('int')
            if len(StimNum) > 4:
                resp_all = np.mean(ff.lp_filter(bf.zscore_CCEP(EEG_resp[rc, StimNum, :]), 45, Fs), 0)
                pk_all = LLf.get_peaks(resp_all)
                p = 1  # polarity of N1, N2 peaks
                if pk_all[0, 1] < pk_all[1, 1]:
                    p = -1
                if np.max(abs(pk_all)) > 5:
                    for cond in np.unique(data[Condition]):
                        StimNum = data.loc[
                            (data[Condition] == cond) & (data.Stim == sc) & (data.Chan == rc), 'Num'].values.astype(
                            'int')
                        resp_raw = np.mean(ff.lp_filter(bf.zscore_CCEP(EEG_resp[rc, StimNum, :]), 45, Fs), 0)
                        peaks = LLf.get_peaks_trial(resp_raw, pk_all[:, 0], p=p, t_0=1, Fs=500)
                        slope = LLf.pk_lin_fit(resp_raw, peaks, fig=0, t_0=1, Fs=500)

                        con_mean.loc[
                            (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc), 'N1'] = abs(
                            peaks[1, 1] - peaks[0, 1])
                        con_mean.loc[
                            (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc), 'N2'] = abs(
                            peaks[1, 1] - peaks[2, 1])
                        con_mean.loc[
                            (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc), 'P2'] = abs(
                            peaks[3, 1] - peaks[2, 1])
                        con_mean.loc[
                            (con_mean[Condition] == cond) & (con_mean.Stim == sc) & (con_mean.Chan == rc), ['sN1',
                                                                                                            'sN2',
                                                                                                            'sP2']] = slope

    return con_mean


def LL_BM_Ph(EEG_resp, stimlist, bad_chans, coord_all, labels_all, StimChanSM, StimChanIx):
    w_LL = 0.25
    t_0 = 1  # time of stimulation in data
    t_Bl = 0.5

    ## calcualte mean CCEP and then take LL
    data_CCEP = np.zeros((1, 5))
    stim_spec = stimlist[(stimlist.IPI_ms == 0) & (stimlist.noise == 0)]  # &(stimlist.noise ==0)
    stimNum = stim_spec.StimNum.values  # [:,0]
    resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
    ChanP1 = SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))
    LL_all = LLf.get_LL_both(data=resps, Fs=Fs, IPI=np.zeros((len(stimNum), 1)), t_0=t_0, win=w_LL)

    # stim_spec0                = stimlist[(stimlist.StimNum.isin((stim_spec.StimNum.values-1)[1:]))]
    # ChanP0                    = np.zeros((len(stimNum),))
    # ChanP0[1:]                = bf.SM2IX(stim_spec0.ChanP.values,StimChanSM,np.array(StimChanIx))

    for c in range(LL_all.shape[0]):
        val = np.zeros((LL_all.shape[1], 5))
        val[:, 0] = c  # response channel
        val[:, 1] = ChanP1  # stim channel
        val[:, 2] = LL_all[c, :, 1]  # LL absolute
        val[:, 3] = stimNum  # stim number in EEG_resp block
        val[:, 4] = stim_spec.condition.values
        val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 2] = np.nan

        # ix         = np.where(np.max(abs(resps[c,:,np.int64(0.95*Fs):np.int64(1.01*Fs)]),1)>400)
        pks = np.max(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1)
        pks_loc = np.argmax(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1) + np.int64(
            (t_0 - 0.05) * Fs)

        ix = np.where((pks > 200) & (pks_loc > np.int64((t_0 - 0.005) * Fs)) & (pks_loc < np.int64((t_0 + 0.012) * Fs)))
        val[ix, 2] = np.nan

        voltage_rec = np.percentile(abs(resps[c, :, 0:np.int64(1 * Fs)]), 90, 1)
        ix = np.where(voltage_rec > 500)
        val[ix, 2] = np.nan
        data_CCEP = np.concatenate((data_CCEP, val), axis=0)

    data_CCEP = data_CCEP[1:-1, :]  # remove first row (dummy row)

    LL_CCEP = pd.DataFrame(
        {"Chan": data_CCEP[:, 0], "Stim": data_CCEP[:, 1], "LL": data_CCEP[:, 2], "zLL": data_CCEP[:, 2],
         "Condition": data_CCEP[:, 4], "Num": data_CCEP[:, 3]})

    LL_CCEP.loc[LL_CCEP['Chan'].isin(bad_chans), 'LL'] = np.nan

    ##Z-score absolute
    LL_BL_z = np.zeros((len(labels_all), 2))
    LL_all_BL = LLf.get_LL_both(data=resps, Fs=Fs, IPI=np.zeros((len(stimNum), 1)), t_0=t_Bl, win=w_LL)
    LL_BL_z[:, 0] = np.nanmean(LL_all_BL[:, :, 1], 1)
    LL_BL_z[:, 1] = np.nanstd(LL_all_BL[:, :, 1], 1)
    for c in range(len(labels_all)):
        LL_CCEP.loc[(LL_CCEP.Chan == c), 'zLL'] = (LL_CCEP.loc[(LL_CCEP.Chan == c), 'LL'] - LL_BL_z[c, 0]) / LL_BL_z[
            c, 1]

    # distance
    for s in np.unique(LL_CCEP.Stim):
        s = np.int64(s)
        for c in np.unique(LL_CCEP.Chan):
            c = np.int64(c)
            LL_CCEP.loc[(LL_CCEP.Stim == s) & (LL_CCEP.Chan == c), 'd'] = math.sqrt(
                ((coord_all[s, 0] - coord_all[c, 0]) ** 2) + ((coord_all[s, 1] - coord_all[c, 1]) ** 2) + (
                        (coord_all[s, 2] - coord_all[c, 2]) ** 2))


def get_LL_thr(EEG_resp, LL_all, labels_all, path_patient, n_trial=3):
    # if 'Num' in LL_all.columns:
    #     LL_all = LL_all.drop('Num', axis=1)
    # LL_all.insert(0, 'Num', LL_all.Num_block)
    ## get threshoold value for each response channel (99th and 95h)
    chan_thr = np.zeros((len(labels_all), 4))  # percentile
    mean_surr = np.zeros((len(labels_all), 200, 2))
    print('Calculating surrogates on mean for each channel')
    for rc in tqdm.tqdm(range(len(labels_all))):
        chan_thr[rc, :], mean_surr[rc, :, :] = get_sig_thr(rc, LL_all, EEG_resp, n_trial, Fs=500, fig_path='no')
    data_A = pd.DataFrame(chan_thr, columns=['99', '95', 'std', 'mean'])
    # data_A.to_csv(path_patient + '/Analysis/BrainMapping/LL/chan_sig_thr.csv', index=False,
    #              header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    # print('Data stored')
    # print(path_patient + '/Analysis/BrainMapping/LL/chan_sig_thr.csv')
    return data_A, chan_thr, mean_surr


def get_sig_thr(rc, LL_CCEP, EEG_resp, t_num, Fs=500, fig_path='no'):
    # t_num = number of trials included for mean calculation, IO =3
    BL_times = np.concatenate(
        [np.arange(0, 0.45, 0.01), np.arange(1.6, 2, 0.01)])  # times wihtout stimulation, 0-0.5s, 1.6 - 2.5
    n = 200  # number of surrogates
    LL_surr = np.zeros((n, 2))
    list_surr = LL_CCEP[(LL_CCEP['d'] > 8) & (LL_CCEP['Chan'] == rc) & ~(LL_CCEP['Stim'] == rc) & ~np.isnan(
        LL_CCEP.LL.values)]  # take BL when rc is not stimulating and not during noise
    # list_surr = list_surr[~np.isnan(list_surr.LL.values)]
    stimNum = list_surr.Num.values.astype('int')
    thr = np.zeros(4, )
    if len(stimNum) > 0:
        for k in range(n):
            t0 = np.random.choice(np.round(BL_times, 2))
            stimNum_choice = np.random.choice(stimNum, t_num)
            EEG_trial = EEG_resp[rc, stimNum_choice,
                        np.int64((t0) * Fs):np.int64((t0 + 0.5) * Fs)]  # np.flip(EEG_resp[rc,stimNum,:],1)
            LL_surr[k, 0], LL_surr[k, 1], _, _ = LL_mx(EEG_trial, t0=0, t1=0.5)

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
    return thr, LL_surr


def LL_mx(EEG_trial, Fs=500, w=0.25, t0=1, t1=1.5, get_P2P=0):
    # calculate mean response and get LL (incl peak)
    resp = ff.lp_filter(np.mean(EEG_trial, 0), 45, Fs)
    LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp, axis=0), 0), Fs, w)

    LL_resp = LL_resp[0, 0]
    mx = np.max(LL_resp[np.int64((t0 + w / 2) * Fs):np.int64((t1 - w / 2) * Fs)])
    mx_ix = np.argmax(LL_resp[np.int64((t0 + w / 2) * Fs):np.int64((t1 - w / 2) * Fs)])
    if get_P2P:
        pN = np.min(resp)
        pP = np.max(resp)
        P2P = np.abs(pP - pN)
    else:
        P2P = 0
    return mx, P2P, mx_ix, LL_resp


def get_SigCon_BM_cond(LL_CCEP, EEG_resp, labels_all, chan_thr):
    # (LL_CCEP[cond_sel]==cond_val)
    Fs = 500
    resp_mean = np.zeros((1, 7))
    for rc in tqdm.tqdm(range(len(labels_all))):  # for each response channel
        for sc in range(len(labels_all)):  # for each stim channel
            for cond_val in np.unique(LL_CCEP.Condition):
                lists = LL_CCEP[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc) & (LL_CCEP.Condition == cond_val)]
                lists = lists[~np.isnan(lists.LL.values)]
                stimNum_all = lists.Num.values.astype('int')

                if len(stimNum_all) > 2:
                    EEG_trial = EEG_resp[rc, stimNum_all[0:3], :]
                    mx, _, _, _ = LL_mx(EEG_trial)

                    resp_z = bf.zscore_CCEP(ff.lp_filter(np.mean(EEG_resp[rc, stimNum_all, :], 0), 45, Fs))
                    z_mx = np.max(abs(resp_z[int(1.01 * Fs):int(1.4 * Fs)]))

                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (LL_CCEP.Condition == cond_val), 'Z_peak'] = z_mx

                    if mx > chan_thr[rc, 0]:
                        LL_CCEP.loc[
                            (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (
                                    LL_CCEP.Condition == cond_val), 'Sig_surr'] = 1
                        LL_CCEP.loc[
                            (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (
                                    LL_CCEP.Condition == cond_val), 'LLpeak'] = mx
                        LL_CCEP.loc[
                            (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (
                                    LL_CCEP.Condition == cond_val), 'LLpeak_z'] = (
                                                                                          mx -
                                                                                          chan_thr[
                                                                                              rc, 3]) / \
                                                                                  chan_thr[
                                                                                      rc, 2]
                    else:
                        LL_CCEP.loc[
                            (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (
                                    LL_CCEP.Condition == cond_val), 'Sig_surr'] = 0
                        LL_CCEP.loc[
                            (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (
                                    LL_CCEP.Condition == cond_val), 'LLpeak'] = mx
                        LL_CCEP.loc[
                            (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (
                                    LL_CCEP.Condition == cond_val), 'LLpeak_z'] = (
                                                                                          mx -
                                                                                          chan_thr[
                                                                                              rc, 3]) / \
                                                                                  chan_thr[
                                                                                      rc, 2]
                else:
                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (LL_CCEP.Condition == cond_val), 'Sig_surr'] = -1
                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (LL_CCEP.Condition == cond_val), 'LLpeak'] = -1
                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc) & (LL_CCEP.Condition == cond_val), 'LLpeak_z'] = -1

    return LL_CCEP


def get_SigCon_mean(LL_CCEP, EEG_resp, labels_all, chan_thr):
    # cond_sel: either Condition or Hour
    M_resp = np.zeros((len(labels_all), len(labels_all), 3)) - 1
    # (LL_CCEP[cond_sel]==cond_val)
    resp_mean = np.zeros((1, 7))
    for rc in tqdm.tqdm(range(len(labels_all))):  # for each response channel
        rc = int(rc)
        for sc in range(len(labels_all)):  # for each stim channel
            lists = LL_CCEP[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc) & (~np.isnan(LL_CCEP.LL.values))]
            d = np.mean(
                LL_CCEP.loc[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc), 'd'])
            lists = lists[~np.isnan(lists.LL.values)]
            stimNum_all = lists.Num.values.astype('int')
            val = np.zeros((1, 7))
            val[0, 0:3] = [rc, sc, d]
            if len(stimNum_all) > 2:
                EEG_trial = EEG_resp[rc, stimNum_all[0:3], :]
                mx, _, _, _ = LL_mx(EEG_trial)

                if mx > chan_thr[rc, 0]:
                    M_resp[sc, rc, 0] = mx
                    M_resp[sc, rc, 1] = 1
                    M_resp[sc, rc, 2] = (mx - chan_thr[rc, 3]) / chan_thr[rc, 2]
                    val[0, 3:6] = [1, mx, (mx - chan_thr[rc, 3]) / chan_thr[rc, 2]]
                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'Sig_block_surr'] = 1
                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'LL_mean'] = mx
                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'LL_mean_z'] = (
                                                                                            mx -
                                                                                            chan_thr[
                                                                                                rc, 3]) / \
                                                                                    chan_thr[
                                                                                        rc, 2]
                else:
                    M_resp[sc, rc, :] = 0
                    val[0, 3:6] = [0, mx, (mx - chan_thr[rc, 3]) / chan_thr[rc, 2]]
                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'Sig_block_surr'] = 0
                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'LL_mean'] = mx
                    LL_CCEP.loc[
                        (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'LL_mean_z'] = (
                                                                                            mx -
                                                                                            chan_thr[
                                                                                                rc, 3]) / \
                                                                                    chan_thr[
                                                                                        rc, 2]
            else:
                M_resp[sc, rc, :] = -1
                val[0, 3:6] = [-1, 0, 0]
                LL_CCEP.loc[(LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'Sig_block_surr'] = -1
                LL_CCEP.loc[
                    (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'LL_mean'] = -1
                LL_CCEP.loc[
                    (LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'LL_mean_z'] = -1
            resp_mean = np.concatenate((resp_mean, val), axis=0)
    resp_mean = resp_mean[1:, :]  # remove first row (dummy row)
    con_mean = pd.DataFrame(
        {"Chan": resp_mean[:, 0], "Stim": resp_mean[:, 1], "d": resp_mean[:, 2], "Sig_block_surr": resp_mean[:, 3],
         "LL": resp_mean[:, 4], "zLL": resp_mean[:, 5]})
    return LL_CCEP, con_mean, M_resp


def plot_BM_Ph(M, labels, areas, c, title, path):
    fig = plt.figure(figsize=(15, 15))
    axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])  # x, y, (start posiion), lenx, leny
    im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=20)
    plt.xlim([-1.5, len(labels) - 0.5])
    plt.ylim([-0.5, len(labels) + 0.5])
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        r = areas[i]
        axmatrix.add_patch(Rectangle((i - 0.5, len(labels) - 0.5), 1, 1, alpha=1,
                                     facecolor=color_regions[np.where(regions == r)[0][0]]))
        axmatrix.add_patch(
            Rectangle((-1.5, i - 0.5), 1, 1, alpha=1, facecolor=color_regions[np.where(regions == r)[0][0]]))
    # Plot colorbar.
    axcolor = fig.add_axes([0.04, 0.85, 0.08, 0.08])  # x, y, x_len, y_len
    circle1 = plt.Circle((0.5, 0.5), 0.4, facecolor=cond_colors[c], edgecolor=[0, 0, 0], alpha=0.6)
    plt.text(0.45, 0.45, titles)
    axcolor.add_patch(circle1)
    plt.axis('off')

    # Plot colorbar.
    axcolor = fig.add_axes([0.9, 0.15, 0.01, 0.7])  # x, y, x_len, y_len
    plt.colorbar(im, cax=axcolor)
    plt.title(title + '-- LL z-score')
    plt.savefig(path + '.jpg')
    plt.savefig(path + '.svg')
    plt.show()


def get_SigCon_BM(LL_CCEP, EEG_resp, labels_all, t_num=3, Fs=500, cond=1):
    n = 200
    M_resp = np.zeros((len(labels_all), len(labels_all), 2)) - 1
    thr = np.zeros((len(labels_all), 4))
    # LL_CCEP.insert(0, 'RespC', 0)
    for rc in tqdm.tqdm(range(len(labels_all))):
        # surr
        BL_times = np.concatenate([np.arange(0, 0.5, 0.01), np.arange(1.6, 2.5, 0.01)])  # times wihtou stimulation
        LL_surr = np.zeros((n, 1))
        list_surr = LL_CCEP[
            (LL_CCEP['Condition'] < 2) & (LL_CCEP['Int'] == 3) & (LL_CCEP['d'] > 8) & (LL_CCEP['Chan'] == rc) & ~(
                    LL_CCEP['Stim'] == rc) & ~np.isnan(LL_CCEP.LL.values)]
        list_surr = list_surr[~np.isnan(list_surr.LL.values)]
        stimNum = list_surr.Num.values.astype('int')
        if len(stimNum) > 0:
            for k in range(n):
                t0 = np.random.choice(np.round(BL_times, 2))
                stimNum_choice = np.random.choice(stimNum, t_num)
                EEG_trial = EEG_resp[rc, stimNum_choice,
                            np.int64((t0) * Fs):np.int64((t0 + 0.4) * Fs)]  # np.flip(EEG_resp[rc,stimNum,:],1)
                LL_surr[k, 0], _, _, _ = LL_mx(EEG_trial, t0=0)

            thr[rc, 0] = np.percentile(LL_surr[:, 0], 99)
            thr[rc, 1] = np.percentile(LL_surr[:, 0], 95)
            thr[rc, 2] = np.percentile(LL_surr[:, 0], 90)
            # thr99 = np.percentile(LL_surr[:,0],99)
            for sc in range(len(labels_all)):
                lists = LL_CCEP[(LL_CCEP['Condition'] < 2) & (LL_CCEP['Int'] == 3) & (LL_CCEP['Chan'] == rc) & (
                        LL_CCEP['Stim'] == sc)]
                lists = lists[~np.isnan(lists.LL.values)]
                stimNum_all = lists.Num.values.astype('int')
                if len(stimNum_all) > 0:
                    EEG_trial = EEG_resp[rc, stimNum_all, :]
                    mx, _, _, _ = LL_mx(EEG_trial)

                    if mx > thr[rc, 0]:
                        M_resp[sc, rc, 0] = mx
                        M_resp[sc, rc, 1] = 1
                        LL_CCEP.loc[(LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'RespC'] = 1
                    else:
                        M_resp[sc, rc, :] = 0
                        LL_CCEP.loc[(LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'RespC'] = 0
                else:
                    LL_CCEP.loc[(LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'RespC'] = -1
        else:
            LL_CCEP.loc[(LL_CCEP.Chan == rc), 'RespC'] = -1
    return LL_CCEP, M_resp, thr


def plot_sigCon_method(EEG_trial, w=0.25, filename='no'):
    mx, _, mx_ix, LL_resp = LL_mx(EEG_trial, w)
    ##########fig
    fig = plt.figure(figsize=(5, 10))  # , facecolor='none'
    fig.subplots_adjust(hspace=0.1, wspace=0)
    plt.suptitle(labels_all[sc] + ' -- ' + labels_all[rc] + ', Distance: ' + str(np.round(lists.d.values[0], 2)) + 'mm')
    gs = fig.add_gridspec(3, 1)  # GridSpec(4,1, height_ratios=[1,2,1,2])
    ax = fig.add_subplot(gs[0, 0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlim([-0.25, 0.75])
    limy = np.max([np.round(np.max(abs(EEG_trial)) / 100) * 100, 200])
    for k in range(len(EEG_trial)):
        plt.plot(x_ax, EEG_trial[k], c=color_elab[0], linewidth=1)
    plt.title('single trials, n:' + str(len(stimNum_all)))
    plt.ylim([-1.1 * limy, 1.1 * limy])
    plt.ylabel('[uV]')
    plt.axvline(0, c=[0, 0, 0])
    plt.xticks([])
    plt.yticks(np.arange(-limy, 1.1 * limy, limy))

    ###mean
    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlim([-0.25, 0.75])
    plt.axvline(0, c=[0, 0, 0])
    plt.plot(x_ax, np.mean(EEG_trial, 0), c=[0, 0, 0], linewidth=3)
    limy = np.max([np.round(np.max(abs(np.mean(EEG_trial, 0))) / 100) * 100, 200])
    plt.ylim([-1.1 * limy, 1.1 * limy])
    plt.yticks(np.arange(-limy, 1.1 * limy, limy))
    plt.axvspan(0.01, w + w / 2, alpha=0.1, color=[1, 0, 0])
    plt.axvspan(0.01 + mx_ix / Fs, 0.01 + mx_ix / Fs + w, alpha=0.2, color=[1, 0, 0])
    plt.title('Mean Response')

    ##### LL
    ax = fig.add_subplot(gs[2, 0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlim([-0.25, 0.75])
    plt.axvline(0, c=[0, 0, 0])
    plt.plot(x_ax, LL_resp, c=[0, 0, 0], linewidth=3)
    plt.title('LL of Mean Response, w=' + str(w) + 's')
    yl = 1.1 * mx
    plt.ylim([0, np.max([1.1 * yl, 3])])
    plt.yticks(np.arange(np.round(np.max([1.1 * yl, 3]))))

    plt.axvspan(0.01 + w / 2, w, alpha=0.1, color=[1, 0, 0])
    plt.plot(0.01 + w / 2 + mx_ix / Fs, mx, 'or')
    plt.ylabel('[uV/ms]')

    if filename != 'no':
        plt.savefig(filename)


def sign_conncetion(LL_CCEP, EEG_resp, labels_all, dur, Fs=500):
    n = 20
    w = 0.1
    win_t = 1 * w

    M_resp = np.zeros((len(labels_all), len(labels_all))) - 1
    LL_CCEP.insert(0, 'RespC', 0)
    for sc in tqdm.tqdm(range(len(labels_all))):
        for rc in range(len(labels_all)):
            lists = LL_CCEP[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
            lists = lists[~np.isnan(lists.LL.values)]
            stimNum_all = lists.Num.values.astype('int')

            if len(stimNum_all) > 0:
                list_BL = LL_CCEP[(LL_CCEP['d'] > 20) & (LL_CCEP['Chan'] == rc) & ~(LL_CCEP['Stim'] == sc) & ~np.isnan(
                    LL_CCEP.LL.values)]
                stimNum_BL = list_BL.Num.values.astype('int')
                resp_BL = np.zeros((n, np.sum(abs(dur)) * Fs))
                for k in range(n):
                    resp_BL[k, :] = ff.lp_filter(
                        np.nanmean(EEG_resp[rc, np.random.choice(stimNum_BL, len(stimNum_all)), :], 0), 45, Fs)
                LL_BL = LLf.get_LL_all(np.expand_dims(resp_BL, axis=0), Fs, w, 1, 0)
                LL_BL = LL_BL[0]

                resp = ff.lp_filter(np.nanmean(EEG_resp[rc, stimNum_all, :], 0), 45, Fs)
                LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp, axis=0), 0), Fs, w, 1, 0)
                LL_resp = LL_resp[0, 0]

                thr = np.percentile(LL_BL[0:np.int64((2.5) * Fs)], 99)

                if all(LL_resp[np.int64((1.01 + w / 2) * Fs):np.int64((1.01 + win_t) * Fs)] > thr):
                    M_resp[sc, rc] = 1
                    LL_CCEP.loc[(LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'RespC'] = 1
                else:
                    M_resp[sc, rc] = 0
                    LL_CCEP.loc[(LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'RespC'] = 0
            else:
                LL_CCEP.loc[(LL_CCEP.Chan == rc) & (LL_CCEP.Stim == sc), 'RespC'] = -1
    return LL_CCEP, M_resp


# def sig_resp(mean, mean_surr, w=0.25, Fs=500):
#     win_t = 1 * w
#     LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(mean, axis=0), 0), Fs, w, 1, 0)
#     LL_resp = LL_resp[0, 0]
#     if len(mean_surr) > 0:
#         LL_BL = LLf.get_LL_all(np.expand_dims(mean_surr, axis=0), Fs, w, 1, 0)
#         LL_BL = LL_BL[0]
#         LL_BL = np.concatenate([LL_BL, np.expand_dims(LL_resp, 0)])
#     else:
#
#         LL_BL = LL_resp
#
#     thr = np.percentile(LL_BL[0:np.int64((0.95 - w / 2) * Fs)], 99)
#
#     if all(LL_resp[np.int64((1.01 + w / 2) * Fs):np.int64((1.01 + win_t) * Fs)] > thr):
#         sig = 1
#     else:
#         sig = 0
#
#     return sig
def dba_cluster(X, n=2):
    km_dba = TimeSeriesKMeans(n_clusters=n, metric="dtw", max_iter=10, max_iter_barycenter=10, random_state=0).fit(X)
    cc = km_dba.cluster_centers_
    y = km_dba.predict(X)
    dist = km_dba.transform(X)
    return cc, y, dist


def get_cluster_pred(sc, rc, LL_CCEP, EEG_resp, Fs=500):
    lists = LL_CCEP[
        ~np.isnan(LL_CCEP.zLL.values) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc) & (LL_CCEP['Condition'] > 0)]
    conds_trials = lists.Condition.values.astype('int')
    stimNum_all = lists.Num.values.astype('int')
    d = 0.5
    trials = EEG_resp[rc, stimNum_all, :]
    trials_z = scipy.stats.zscore(trials, 1)
    data = trials_z[:, np.int64(1 * Fs):np.int64((1 + d) * Fs)]
    pred_loss = np.zeros((2,)) - 1
    if data.shape[0] > 0:
        cc, y_pred, dist = dba_cluster(np.expand_dims(data, -1))
        D = dist / np.max(dist)

        i = 0
        for cond in np.unique(conds_trials):
            d = 0
            for x, y in itertools.combinations(y_pred[conds_trials == cond], 2):
                d += np.square(x - y)
            pred_loss[i] = d  # np.sqrt(d)
            i = 1 + i

    return pred_loss


def get_mean_BL_B(sc, rc, EEG_resp, LL_CCEP, conds=[1, 3], w=0.25, Fs=500):
    rc = np.int64(rc)
    sc = np.int64(sc)
    lists = LL_CCEP[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
    lists = lists[~np.isnan(lists.zLL.values)]
    sig = 1
    k = 1
    n = 3
    EEG = np.zeros((1, 2000))
    for i in range(len(conds)):
        cs = conds[i]
        list_cond = lists[lists.Condition == cs]
        stimNum = list_cond.Num.values.astype('int')
        if len(stimNum) < n:
            n = len(stimNum)
        if cs == 3:
            k = -1
        EEG = np.concatenate([EEG, k * EEG_resp[rc, stimNum[0:n], :]])
    EEG = EEG[1:, :]
    resp = np.nanmean(EEG[:, :], 0)
    LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp, axis=0), 0), Fs, w, 1, 0)
    LL_resp = LL_resp[0, 0]
    thr = np.percentile(LL_resp[0:np.int64((1 - w) * Fs)], 99)
    # if  all(LL_resp[np.int64((1.01+w/2)*Fs):np.int64((1.01+w)*Fs)]>thr):
    #     sig = 2
    if np.max(LL_resp[np.int64((1.01 + w / 2) * Fs):np.int64((1.01 + w) * Fs)]) > thr:
        sig = 2
    return sig