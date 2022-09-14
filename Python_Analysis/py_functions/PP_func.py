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
from scipy.spatial import distance
from numpy import trapz

# regions         = pd.read_excel("T:\EL_experiment\Patients\\" +'all'+"\elab_labels.xlsx", sheet_name='regions', header=0)
# color_regions   = regions.color.values
# regions         = regions.label.values
cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]

# data normally in 4s locks, [-1,3]
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


def get_IPI_switch(IPI_all, IPI_selected):
    mn = 0
    mx = 0
    for ix in np.where(np.isin(IPI_all, IPI_selected))[0]:
        if np.mean(np.isin(IPI_all, IPI_selected)[ix:ix + 4]) > 0.7:
            mn = IPI_all[ix]
            break
    for ix in np.where(np.isin(IPI_all, IPI_selected))[0]:
        if np.mean(np.isin(IPI_all, IPI_selected)[ix - 3:ix + 1]) > 0.7:
            mx = IPI_all[ix]
    return [mn, mx]


def get_LL_all_cond(EEG_resp, stimlist, lbls, bad_chans, w=0.25, Fs=500, t_0=1):
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist,
        lbls)
    data_LL = np.zeros((1, 13))  # RespChan, Int, LL, LLnorm, State
    Int_prob = 2
    stim_spec = stimlist[(stimlist.noise == 0)]  # &(stimlist.noise ==0)
    if len(stim_spec)>0:
        stimNum = stim_spec.StimNum.values  # [:,0]
        # resps = EEG_resp[:, stimNum, :]
        resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
        ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))
        IPIs = np.expand_dims(np.array(stim_spec.IPI_ms.values), 1)
        LL = LLf.get_LL_both(data=resps, Fs=Fs, IPI=IPIs, t_0=1, win=w)
        # todo: change to depending on IPI
        # second pulse
        w_start = np.int64(
            np.round((IPIs / 1000 + t_0 + 0.01) * Fs))  # start position at sencond trigger plus 20ms (art removal)
        w_end = np.int64(np.round((IPIs / 1000 + t_0 + 0.5) * Fs) - 1)
        n = np.int64((w_end - w_start)[0, 0])
        inds = np.linspace(w_start, w_end, n).T.astype(int)
        resp_PP = np.take_along_axis(resps, inds, axis=2)
        # LL_trial = LLf.get_LL_all(resps[:, :, int(1 * Fs):int(1.5 * Fs)], Fs, 0.25, t_0, IPIs)
        # LL_peak = np.max(LL_trial, 2)
        LL_trial = LLf.get_LL_all(resp_PP, Fs, 0.25, t_0, IPIs)
        LL_peakPP = np.max(LL_trial, 2)
        pk_start = 0.5
        for c in range(len(LL)):
            val = np.zeros((LL.shape[1], 13))
            val[:, 0] = c  # response channel
            val[:, 1] = ChanP1
            val[:, 4] = stim_spec.Int_cond.values  # Intensity
            val[:, 2] = LL[c, :, 1]  # PP
            #val[:, 3] = LL[c, :, 0]  # SP
            val[:, 11] = stim_spec.IPI_ms.values
            val[:, 3] = LL_peakPP[c, :]  # SP
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
        # add SP 2mA
        stim_spec = stimlist[(stimlist.IPI_ms > 500)&(stimlist.Int_cond == Int_prob)&(stimlist.noise == 0)]  # &(stimlist.noise ==0)
        stimNum = stim_spec.StimNum.values  # [:,0]
        resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
        ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))
        IPIs = np.expand_dims(np.array(stim_spec.IPI_ms.values), 1)
        IPIs = np.zeros_like(IPIs)
        LL = LLf.get_LL_both(data=resps, Fs=Fs, IPI=IPIs, t_0=1, win=w)
        # todo: change to depending on IPI
        # second pulse
        LL_trial = LLf.get_LL_all(resps[:, :, int(1 * Fs):int(1.5 * Fs)], Fs, 0.25, t_0, IPIs)
        LL_peak = np.max(LL_trial, 2)

        pk_start = 0.5
        for c in range(len(LL)):
            val = np.zeros((LL.shape[1], 13))
            val[:, 0] = c  # response channel
            val[:, 1] = ChanP1
            val[:, 4] = 0
            val[:, 2] = LL[c, :, 1]  # PP
            # val[:, 3] = LL[c, :, 0]  # SP
            val[:, 11] = 0
            val[:, 3] = LL_peak[c, :]  # SP
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
            {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "LL_peak": data_LL[:, 3], "Int": data_LL[:, 4],
             'IPI':data_LL[:, 11], 'Condition': data_LL[:, 5], 'Hour': data_LL[:, 6], "Block": data_LL[:, 10], "Sleep": data_LL[:, 9],
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
        LL_all.loc[(LL_all.Chan).isin(bad_chans), 'LL_peak'] = np.nan
        #file = path_patient + '/Analysis/PairedPulse/' + cond_folder + '/data/con_trial.csv'
        #LL_all.to_csv(file, index=False, header=True)  # scat_plot = scat_plot.fillna(method='ffill')
        #print(file + ' -- stored')
    else:
        data_LL = np.zeros((1,12))
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "LL_peak": data_LL[:, 3],
             "Int": data_LL[:, 4],
             'IPI': data_LL[:, 11], 'Condition': data_LL[:, 5], 'Hour': data_LL[:, 6], "Block": data_LL[:, 10],
             "Sleep": data_LL[:, 9],
             "Num_block": data_LL[:, 7], "Date": data_LL[:, 8]})
    return LL_all


def get_LL_all_Ph(stimlist, EEG_resp, Stims, labels_clinic, coord_all, path_patient, w=0.25):
    StimChanSM = np.unique(stimlist.ChanP)

    data_LL = np.zeros((1, 10))  # RespChan, Int, LL, LLnorm, State
    stim_spec = stimlist[(stimlist.condition > 0) & (stimlist.noise == 0)]  # &(stimlist.noise ==0)
    stimNum = stim_spec.StimNum.values  # [:,0]
    resps = EEG_resp[:, stimNum, :]
    ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(Stims))
    IPIs = np.expand_dims(np.array(stim_spec.IPI_ms.values), 1)
    LL = LL_funcs.get_LL_both(data=resps, Fs=Fs, IPI=IPIs, t_0=1, win=w)
    for c in range(len(LL)):
        val = np.zeros((LL.shape[1], 10))
        val[:, 0] = c  # response channel
        val[:, 1] = ChanP1
        val[:, 4] = stim_spec.Int_cond.values  # Intensity
        val[:, 2] = LL[c, :, 1]  # PP
        val[:, 3] = LL[c, :, 0]  # SP
        val[:, 5] = stim_spec.IPI_ms.values
        val[:, 6] = stim_spec.condition.values
        val[:, 7] = stimNum
        val[:, 8] = LL[c, :, 1] / LL[c, :, 0]
        val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 2:4] = np.nan
        val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 8] = np.nan

        data_LL = np.concatenate((data_LL, val), axis=0)

    data_LL = data_LL[1:-1, :]  # remove first row (dummy row)
    LL_all = pd.DataFrame(
        {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "LL SP": data_LL[:, 3],
         "rLL": data_LL[:, 8], "nLL": data_LL[:, 2], "Int": data_LL[:, 4], "IPI": data_LL[:, 5],
         "Condition": data_LL[:, 6], "Num": data_LL[:, 7]})
    # define condition (benzo, bl, sleepdep)
    LL_all.insert(7, "Cond", 0, True)
    for j in range(len(cond_vals)):
        LL_all.loc[(LL_all.Condition == cond_vals[j]), 'Cond'] = cond_labels[j]

    # distance
    s_i = 0
    for s in np.unique(LL_all.Stim):
        s = np.int64(s)
        for c in np.unique(LL_all.Chan):
            c = np.int64(c)
            LL_all.loc[(LL_all.Stim == s) & (LL_all.Chan == c), 'd'] = distance.euclidean(coord_all[s], coord_all[c])
            LL_all.loc[(LL_all.Stim == s) & (LL_all.Chan == c), 'nLL'] = LL_all.loc[(LL_all.Stim == s) & (
                    LL_all.Chan == c), 'LL'] / np.nanmean(LL_all.loc[(LL_all.Stim == s) & (LL_all.Chan == c) & (
                    LL_all.Int == 2) & (LL_all.Condition == 1), 'LL SP'])
        s_i = s_i + 1
    LL_all = LL_all[~np.isnan(LL_all.LL)]
    LL_all = LL_all.reset_index(drop=True)
    LL_all.loc[LL_all.Int == 0, 'LL SP'] = LL_all.loc[LL_all.Int == 0, 'LL']
    LL_all.to_csv(path_patient + '/Analysis/PairedPulse/Ph/data/LL_all.csv', index=False,
                  header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    return LL_all


def get_LL_thr(EEG_resp, LL_all, labels_all, path_patient, n=3):
    ## get threshoold value for each response channel (99th and 95h)
    chan_thr = np.zeros((len(labels_all), 4))
    for rc in range(len(labels_all)):
        chan_thr[rc, :] = get_sig_thr(rc, LL_all, EEG_resp, n)
    data_A = pd.DataFrame(chan_thr, columns=['99', '95', '90', 'std'])
    data_A.to_csv(path_patient + '/Analysis/PairedPulse/Ph/data/chan_sig_thr.csv', index=False,
                  header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    return chan_thr


# tobe saved in basic functions
def LL_mx(EEG_trial, w=0.25, t0=1.01):
    # calculate mean response and get LL (incl peak)
    resp = ff.lp_filter(np.mean(EEG_trial, 0), 40, Fs)
    LL_resp = LL_funcs.get_LL_all(np.expand_dims(np.expand_dims(resp, axis=0), 0), Fs, w, 1, 0)
    LL_resp = LL_resp[0, 0]
    mx = np.max(LL_resp[np.int64((t0 + w / 2) * Fs):np.int64((t0 + 1.5 * w) * Fs)])
    mx_ix = np.argmax(LL_resp[np.int64((t0 + w / 2) * Fs):np.int64((t0 + 1.5 * w) * Fs)])
    return mx, mx_ix, LL_resp


def sig_resp(mean, thr, w=0.25):
    # check whether a mean response is a significant CCEP based on a pre-calculated threshold thr
    mean = ff.lp_filter(mean, 40, Fs)
    LL_resp = LL_funcs.get_LL_all(np.expand_dims(np.expand_dims(mean, axis=0), 0), Fs, w, 1, 0)
    LL_resp = LL_resp[0, 0]
    mx = np.max(LL_resp[np.int64((1.01 + w / 2) * Fs):np.int64((1.01 + 1.5 * w) * Fs)])
    max_ix = np.argmax(LL_resp[np.int64((1.01 + w / 2) * Fs):np.int64((1.01 + 1.5 * w) * Fs)])
    if mx > thr:
        sig = 1
    else:
        sig = 0
    return LL_resp, mx, max_ix, sig


def get_sig_thr(rc, LL_CCEP, EEG_resp, t_num, w=0.25):
    BL_times = np.concatenate(
        [np.arange(0, 1 - 2 * w - 0.05, 0.01), np.arange(2.3, 3 - 2 * w, 0.01)])  # times wihtou stimulation
    n = 200  # number of surrogates
    LL_surr = np.zeros((n, 1))
    list_surr = LL_CCEP[(LL_CCEP['Chan'] == rc) & ~(LL_CCEP['Stim'] == rc) & ~np.isnan(LL_CCEP.LL.values)]
    list_surr = list_surr[~np.isnan(list_surr.LL.values)]
    stimNum = list_surr.Num.values.astype('int')
    thr = np.zeros(4, )
    if len(stimNum) > 0:
        for k in range(n):
            t0 = np.random.choice(np.round(BL_times, 2))
            stimNum_choice = np.random.choice(stimNum, t_num)
            EEG_trial = EEG_resp[rc, stimNum_choice,
                        np.int64((t0) * Fs):np.int64((t0 + 2 * w) * Fs)]  # np.flip(EEG_resp[rc,stimNum,:],1)
            LL_surr[k, 0], _, _ = LL_mx(EEG_trial, t0=0)

        thr[0] = np.percentile(LL_surr[:, 0], 99)
        thr[1] = np.percentile(LL_surr[:, 0], 95)
        thr[2] = np.percentile(LL_surr[:, 0], 90)
        thr[3] = np.mean(LL_surr[:, 0]) + np.std(LL_surr[:, 0])
        # fig = plt.figure(figsize=(5,5))
        # plt.title('surrogates - '+labels_all[rc])
        # plt.hist(LL_surr[:,0])
        # plt.axvline(thr[0], c= [1,0,0], label='99%')
        # plt.axvline(thr[1], c= [1,0,0], label='90%')
        # plt.axvline(np.mean(LL_surr[:,0])+np.std(LL_surr[:,0]), c= [0,0,0], label='mean +std')
        # plt.xlabel('LL [250ms]')
        # plt.xlim([0,np.max([2,1.1*max(LL_surr[:,0])]) ])
        # plt.legend()
        # plt.savefig(path_patient + '/Analysis/PairedPulse/Ph/figures/surr/'+subj+'_surr_LL_'+labels_all[rc]+'.jpg')
        # plt.close(fig)    # close the figure window
    return thr


def get_sig_Con(EEG_resp, LL_all, chan_thr, path_patient, labels_all, w=0.25, Int_prob=2):
    subj = path_patient[-5:]
    Stims = np.unique(LL_all.Stim)
    sig_probing = np.zeros((len(EEG_resp), len(Stims)))

    for i in range(0, len(Stims)):
        sc = Stims[i].astype('int')
        SP_data = LL_all[(LL_all['Int'] >= Int_prob) & (LL_all['Stim'] == sc) & (LL_all['IPI'] > 2 * w * 1000) & (
                    LL_all['Condition'] == 1)]
        fig, axs = plt.subplots(5, np.ceil(len(EEG_resp) / 5).astype('int'), figsize=(20, 10))
        axs = axs.reshape(-1)
        plt.suptitle(subj + ' -- Stim: ' + labels_all[sc.astype('int')])
        for rc in np.unique(SP_data.Chan):
            rc = np.int64(rc)
            thr = chan_thr[rc, 0]
            mean = np.nanmean(EEG_resp[rc, np.unique(SP_data.Num.values.astype('int')), :], 0)
            LL_resp, mx, max_ix, sig = sig_resp(mean, thr, w)
            axs[rc].plot(x_ax, ff.lp_filter(mean, 40, Fs), c=[0, 0, sig], alpha=0.5 + 0.5 * sig)
            axs[rc].set_xlim([-0.1, 0.3])
            axs[rc].set_ylim([-800, 800])
            axs[rc].set_title(labels_all[rc])
            axs[rc].axvline(0, c=[1, 0, 0])

            sig_probing[rc, i] = sig
            LL_all.loc[(LL_all.Stim == sc) & (LL_all.Chan == rc), 'Sig_Con'] = sig
        for rc in range(5 * np.ceil(len(EEG_resp) / 5).astype('int')):
            axs[rc].axis('off')
        plt.savefig(path_patient + '/Analysis/PairedPulse/Ph/figures/' + subj + '_' + labels_all[sc] + '_sigCon.jpg')
        plt.close()
    data_A = pd.DataFrame(sig_probing, columns=[labels_all[Stims[0].astype('int')], labels_all[Stims[1].astype('int')]])
    data_A.to_csv(path_patient + '/Analysis/PairedPulse/Ph/data/sig_CCEP.csv', index=False,
                  header=False)  # scat_plot = scat_plot.fillna(method='ffill')

    if not 'Effect' in LL_all.columns:
        LL_all.insert(8, 'Effect', 0)
    for i in range(0, len(Stims)):
        sc = Stims[i].astype('int')
        for rc in np.unique(LL_all.loc[(LL_all['Sig_Con'] == 1) & (LL_all['Stim'] == sc), 'Chan']).astype('int'):
            SP_data = LL_all[(LL_all['Chan'] == rc) & ((LL_all['Int'] == Int_prob) | (LL_all['Int'] == 0)) & (
                        LL_all['Stim'] == sc) & ((
                                                         LL_all['IPI'] > w * 1000) | (LL_all['IPI'] == 0)) & (
                                         LL_all['Condition'] == 1)]
            thr_sf = [np.percentile(SP_data['LL SP'], 5), np.percentile(SP_data['LL SP'], 95)]
            LL_all.loc[(LL_all['Chan'] == rc) & (LL_all['Stim'] == sc) & (LL_all['LL'] < thr_sf[0]), 'Effect'] = -1
            LL_all.loc[(LL_all['Chan'] == rc) & (LL_all['Stim'] == sc) & (LL_all['LL'] > thr_sf[1]), 'Effect'] = 1
    LL_all.to_csv(path_patient + '/Analysis/PairedPulse/Ph/data/LL_all.csv', index=False,
                  header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    return LL_all


def get_PP_summary(LL_all, path_patient):
    # get IPI which are inducing suppresion / facilitation for each connection and condition
    IPI_all = np.unique(LL_all.IPI)
    Stims = np.unique(LL_all.Stim)
    Int_all = np.unique(LL_all.Int)
    PP_sum = np.zeros((1, 11))
    for i in range(0, len(Stims)):
        sc = Stims[i].astype('int')
        for rc in np.unique(LL_all.loc[(LL_all['Sig_Con'] == 1) & (LL_all['Stim'] == sc), 'Chan']).astype('int'):
            for Int in Int_all:
                for cond in [1, 3]:
                    PP_sum1 = np.zeros((1, 11))
                    data = LL_all[(LL_all['Condition'] == cond) & (LL_all['Int'] == Int) & (LL_all['Stim'] == sc) & (
                            LL_all['Chan'] == rc) & (LL_all['Effect'] == 1)]
                    IPI_fac = np.unique(data.IPI)
                    data = LL_all[(LL_all['Condition'] == cond) & (LL_all['Int'] == Int) & (LL_all['Stim'] == sc) & (
                            LL_all['Chan'] == rc) & (LL_all['Effect'] == -1)]
                    IPI_sup = np.unique(data.IPI)
                    IPI_sup = get_IPI_switch(IPI_all, IPI_sup)
                    IPI_fac = get_IPI_switch(IPI_all, IPI_fac)
                    PP_sum1[0, 0] = sc
                    PP_sum1[0, 1] = rc
                    PP_sum1[0, 2] = cond
                    PP_sum1[0, 3:5] = np.array(IPI_sup)
                    PP_sum1[0, 5:7] = np.array(IPI_fac)
                    PP_sum1[0, 7] = np.mean(LL_all.loc[(LL_all['Condition'] == cond) & (LL_all['Int'] == Int) & (
                            LL_all['Stim'] == sc) & (LL_all['Chan'] == rc) & (LL_all['IPI'] >= IPI_sup[0]) & (
                                                               LL_all['IPI'] <= IPI_sup[1]), 'nLL'])
                    PP_sum1[0, 8] = np.mean(LL_all.loc[(LL_all['Condition'] == cond) & (LL_all['Int'] == Int) & (
                            LL_all['Stim'] == sc) & (LL_all['Chan'] == rc) & (LL_all['IPI'] >= IPI_fac[0]) & (
                                                               LL_all['IPI'] <= IPI_fac[1]), 'nLL'])
                    PP_sum1[0, 9] = Int
                    PP_sum1[0, 10] = np.round(np.mean(LL_all.loc[(LL_all['Stim'] == sc) & (
                            LL_all['Chan'] == rc), 'd']), 2)
                    PP_sum = np.concatenate([PP_sum, PP_sum1], 0)
    PP_sum = PP_sum[1:, :]
    PP_sum = pd.DataFrame(PP_sum,
                          columns=['Stim', 'Chan', 'Condition', 'Sup0', 'Sup1', 'Fac0', 'Fac1', 'LLsup', 'LLfac', 'Int',
                                   'd'])
    PP_sum.to_csv(path_patient + '/Analysis/PairedPulse/Ph/data/PP_sum.csv', index=False,
                  header=True)
    return PP_sum


def plot_PP_cond(sc, rc, stimNum, LL_CCEP, EEG_resp):
    listsT = LL_CCEP[(LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc) & (LL_CCEP['Num'] == stimNum)]
    IPI = listsT.IPI.values[0]
    Int = listsT.Int.values[0]

    lists = LL_CCEP[
        (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc) & (LL_CCEP['IPI'] == IPI) & (LL_CCEP['Int'] == Int)]
    stimNum_all = lists.Num.values.astype('int')
    limy = 1.5 * np.max(abs(EEG_resp[rc, stimNum_all, :]))
    print(limy)
    lists_SP = LL_CCEP[
        (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc) & (LL_CCEP['Int'] == 2) & (LL_CCEP['IPI'] > 300)]
    stimNum_SP = lists_SP.Num.values.astype('int')
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3])  # GridSpec(4,1, height_ratios=[1,2,1,2])
    fig.subplots_adjust(hspace=.1, wspace=.1)
    plt.suptitle(labels_all[sc] + ' -- ' + labels_all[rc] + ', Dist: ' + str(np.round(lists.d.values[0])) + 'mm')

    ax = fig.add_subplot(gs[0, 0])
    ax.axvspan(0, 0.25, facecolor=color_elab[1], alpha=0.5)
    plt.ylim([-np.max([limy, 400]), np.max([limy, 400])])
    plt.axvline(0, c=[1, 0, 0])
    plt.title('mean 2mA SP')
    plt.xticks([0, 0.25])
    plt.plot(x_ax, np.mean(EEG_resp[rc, stimNum_SP, :], 0), c=[0, 0, 0], linewidth=2)
    # plt.title(labels_all[Stim_chs]+' -- '+labels_clinic[rc])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlim([-0.1, 0.5])

    ax = fig.add_subplot(gs[0, 1])
    ax.axvspan(IPI / 1000, IPI / 1000 + 0.25, facecolor=color_elab[1], alpha=0.5)
    plt.ylim([-np.max([limy * 1.1, 400]), np.max([limy * 1.1, 400])])
    plt.xlim([-0.05, 1.85])
    plt.ylim([-np.max([limy, 400]), np.max([limy, 400])])
    plt.axvline(0, c=[1, 0, 0])
    plt.axvline(IPI / 1000, c=[1, 0, 0])
    # plt.axvline(IPI/1000+0.25, c=[0,0,0])

    plt.xlabel('time [s]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    conds = lists.Condition.values.astype('int')
    nLLs = lists.nLL.values
    for i in range(len(stimNum_all)):
        plt.plot(x_ax, ff.lp_filter(EEG_resp[rc, stimNum_all[i], :], 45, Fs), c=cond_colors[np.int64(conds[i])],
                 linewidth=2, label=cond_labels[np.int64(conds[i])])
        plt.text(IPI / 1000 + 0.3, 200 + i * 100, 'nLL: ' + str(np.round(nLLs[i], 3)),
                 c=cond_colors[np.int64(conds[i])])
    plt.legend()
    plt.text(-0.03, 410, str(Int) + 'mA')
    plt.yticks([])
    plt.xticks([0, IPI / 1000, 0.5, 1, 1.5])
    plt.show()
