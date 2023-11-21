import os
import numpy as np
import pandas as pd
import basic_func as bf
import freq_funcs as ff
import LL_funcs as LLf
import Cluster_func as Cf
from scipy.spatial import distance
import scipy

import significance_funcs as sf

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def get_GT_trials(trials, real=1, t_0=1, Fs=500, w=0.25, n_cluster=2, cluster_type='Kmeans'):
    # t_0 stimulation time in epoched data, changed in surrogate data
    # trials = EEG_resp[rc, stimNum_all, :], shape (n,2000)
    # 1. z-score trials and get LL for t_onset and t_max
    EEG_trial = ff.bp_filter(trials, 1, 45, Fs)
    # EEG_trial = stats.zscore(EEG_trial, axis=1)
    resp_mean = np.nanmean(EEG_trial, 0)  ## ff.lp_filter(np.nanmean(bf.zscore_CCEP(trials), 0), 40, Fs)
    LL_mean = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp_mean, 0), 0), Fs, w)
    # where LL is the highest
    t_WOI = np.argmax(LL_mean[0, 0, int((t_0 + w / 2) * Fs):int((t_0 + 0.5 - w / 2) * Fs)]) / Fs
    if real:
        thr = np.percentile(np.concatenate([LL_mean[0, 0, int((w / 2) * Fs):int((t_0 - w / 2) * Fs)],
                                            LL_mean[0, 0, int(3 * Fs):int((4 - w / 2) * Fs)]]),
                            99)  # LL_resp[0, 0, int((t_0+0.5) * Fs):] = 0 * Fs):] = 0
        LL_t = np.array(LL_mean[0, 0, :int((t_0 + 0.5) * Fs)] > thr) * 1
        t_resp_all = sf.search_sequence_numpy(LL_t, np.ones((int((w + 0.08) * Fs),)))
        if len(t_resp_all) > 0:
            t_onset = t_resp_all[0] / Fs - t_0 + w / 2
            r = 1
            if (t_onset < 0.001) | (t_onset > 0.5):
                t_onset = 0
        else:
            r = 0
            t_onset = 0
    else:  # surr
        r = 10
        t_onset = 0
        thr = 0
    # cluster trials based on specific window (where highest LL is, WOI) on zs-cored data (not affectd by amplitude)
    # EEG_trial = stats.zscore(EEG_trial, axis=1)
    if cluster_type == 'Kmeans':
        EEG_trial = bf.zscore_CCEP(EEG_trial, t_0, Fs)
        cc, y = Cf.ts_cluster(
            EEG_trial[:, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + w) * Fs)], n=n_cluster,
            method='euclidean')
    else:  # similarity measure
        # EEG_trial = bf.zscore_CCEP(EEG_trial, t_0, Fs)
        cc, y = Cf.ts_cluster(
            EEG_trial[:, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + w) * Fs)], n=n_cluster,
            method='similarity')
    # p_CC = np.corrcoef(cc[0], cc[1])[0, 1]  # pearson correlation of two clusters

    ## store mean across all trials and mean for specific cluster (CC)
    M_GT = np.zeros((3, trials.shape[-1]))
    M_GT[0, :] = ff.lp_filter(np.nanmean(trials, 0), 45, Fs)
    for c in range(n_cluster):
        M_GT[c + 1, :] = ff.lp_filter(np.nanmean(trials[y == c, :], 0), 45, Fs)

    # pearson correlation of the two CC
    p_CC = np.corrcoef(M_GT[1, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + 2 * w) * Fs)],
                       M_GT[2, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + 2 * w) * Fs)])[0, 1]

    # todo: LL of WOI or LL max
    LL_CC = LLf.get_LL_all(np.expand_dims(M_GT[1:, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + w) * Fs)], 0), Fs, w)
    LL_CC = np.max(LL_CC[0, :, :], 1)
    LL_mean_max = np.max(LL_mean[0, 0, int((t_0 + w / 2) * Fs):int((t_0 + 0.5 - w / 2) * Fs)])
    return [r, t_onset, t_WOI], LL_mean[0, 0], p_CC, M_GT, y, thr, LL_CC, LL_mean_max


def get_GT(sc, rc, LL_CCEP, EEG_resp, Fs=500, t_0=1, w_cluster=0.25, n_cluster=2):
    # for each connection (A-B) the two CC are calculated
    lists = LL_CCEP[(LL_CCEP['Ictal'] == 0) &(LL_CCEP['Artefact'] < 1) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
    stimNum_all = lists.Num.values.astype('int')
    M_GT = np.zeros((n_cluster + 1, 2000))
    r = -1
    t_onset = -1
    t_WOI = -1
    p_CC = -1
    LL_mean_max = -1
    LL_CC = np.zeros((1, 2))
    if len(stimNum_all) > 0:
        t_nan = np.where(np.isnan(np.mean(EEG_resp[rc, stimNum_all, :], 1)) * 1)  # [0][0]
        if len(t_nan[0]) > 0:
            stimNum_all = np.delete(stimNum_all, t_nan[0])
        if len(stimNum_all) > 15:
            trials = EEG_resp[rc, stimNum_all, :]
            [r, t_onset, t_WOI], _, _, M_GT, y, _, LL_CC,LL_mean_max = get_GT_trials(trials, real=1, t_0=t_0, Fs=Fs,
                                                                         w=w_cluster,
                                                                         n_cluster=n_cluster, cluster_type='similarity')
    return M_GT, [r, t_onset, t_WOI], LL_CC, LL_mean_max


def get_CC_surr(rc, LL_CCEP, EEG_resp, n_trials, Fs=500, w_cluster=0.25, n_cluster=2, n_surr=200):
    # general infos
    # non-stim surrogate trials
    # other trials
    stim_trials = np.unique(
        LL_CCEP.loc[(LL_CCEP.Stim >= rc - 1) & (LL_CCEP.Stim <= rc + 1), 'Num'].values.astype('int'))
    StimNum = np.unique(LL_CCEP.loc[(LL_CCEP.d > 15) &(LL_CCEP['Ictal'] == 0) & (LL_CCEP.Artefact == 0) & (
            LL_CCEP.Chan == rc), 'Num'])  # np.random.choice(np.unique(con_trial.Num), size=400)
    StimNum = [i for i in StimNum if i not in stim_trials]
    StimNum = [i for i in StimNum if i not in stim_trials + 1]
    StimNum = [i for i in StimNum if i not in stim_trials - 1]

    # initiate
    # pear_surr = np.zeros((1,)) - 1
    # epoch: [-1,3]s: switch [-1,1] and [1,3] to have non stimulating data after onset at 0
    LL_mean_surr = np.zeros((n_surr, 1))
    LL_CC_surr = np.zeros((n_surr, 2))
    WOI_surr = np.zeros((n_surr,))
    LL_surr_data = np.zeros((n_surr, 2, 2000))
    for i in range(n_surr):
        StimNum_surr = np.unique(np.random.choice(StimNum, size=n_trials).astype('int'))
        t_nan = np.where(np.isnan(np.mean(EEG_resp[rc, StimNum_surr, :], 1)) * 1)  # [0][0]
        if len(t_nan[0]) > 0:
            StimNum_surr = np.delete(StimNum_surr, t_nan[0])
        # get overall mean and two clusters
        trials = EEG_resp[rc, StimNum_surr, :]
        for t_0_surr in [0.2]:
            [_, _, t_WOI], _, p_CC, M_GT, y, _, LL_CC,LL_mean_max = get_GT_trials(trials, real=0, t_0=t_0_surr, Fs=Fs, w=w_cluster,
                                                                            n_cluster=n_cluster,
                                                                            cluster_type='similarity')
            LL_mean_surr[i] = LL_mean_max
            # return [r, t_onset, t_WOI], LL_mean[0, 0], p_CC, M_GT, y, thr, LL_CC
            if (sum(y == 0) > np.max([5, 0.05 * len(y)])) & (sum(y == 1) > np.max([5, 0.05 * len(y)])):
                # pear_surr = np.concatenate([pear_surr, [p_CC]], 0)
                LL_CC_surr[i] = LL_CC
                WOI_surr[i] = t_WOI
                LL_surr_data[i] = M_GT[1:, :]
            else:
                LL_CC_surr[i, :] = np.nan
                WOI_surr[i] = np.nan
                LL_surr_data[i, 0, :] = np.nan
                LL_surr_data[i, 1, :] = np.nan
    return LL_CC_surr, LL_surr_data, WOI_surr, LL_mean_surr
    # return pear_surr, [np.percentile(pear_surr, 95), np.percentile(pear_surr, 99)], LL_surr[1:, :]


def get_CC_summ(M_GT_all, M_t_resp, surr_thr, coord_all, t_0=1, w=0.25, w_LL_onset=0.1, smooth_win=0.1, Fs=500):
    # creates a table for each stim-chan pair with the two CC found indicating the WOI, LL and whether it's signficant
    start = 1

    smooth_win = int(smooth_win * Fs)
    if np.mod(smooth_win, 2) == 0:
        smooth_win = smooth_win + 1

    for sc in range(M_GT_all.shape[0]):
        for rc in range(M_GT_all.shape[0]):
            d = np.round(distance.euclidean(coord_all[sc], coord_all[rc]), 2)
            LL_CC = LLf.get_LL_all(np.expand_dims(M_GT_all[sc, rc, :], 0), Fs, 0.25)[0]
            WOI = M_t_resp[sc, rc, 2]
            if M_t_resp[sc, rc, 0] > -1:
                thr = surr_thr.loc[surr_thr.Chan == rc, 'CC_LL95'].values[0]
                thr50 = surr_thr.loc[surr_thr.Chan == rc, 'CC_LL50'].values[0]
                thr50 = thr50 + (thr - thr50) / 2
                sig_mean = np.array(M_t_resp[sc, rc, 5]>surr_thr.loc[surr_thr.Chan == rc, 'mean_LL95'].values[0])*0.5
                for i in range(1, 3):
                    LL_peak = np.max(LL_CC[i, int((t_0 + w / 2) * Fs):int((t_0 + 0.5 - w / 2) * Fs)])
                    LL_WOI = LL_CC[i, int((t_0 + WOI + w / 2) * Fs)]
                    sig = np.array(LL_WOI > thr) * 1
                    art = 0
                    thr_BL = np.percentile(LL_CC[i, :int((t_0 - w / 2) * Fs)], 95)
                    sig_BL = np.array(LL_WOI > thr_BL) * 1
                    # if sig: # only if CC surpasse LL threshold
                    #     thr_BL = np.percentile(LL_CC[i,:int((t_0 - w / 2) * Fs)],95)
                    #     #### Check whether LL surpasses the threshold for some time (more specific test)
                    #     LL_t_pk = np.array(LL_CC[i] >= np.percentile(LL_CC[i,:int((t_0 - w / 2) * Fs)],95)) * 1
                    #     LL_t_pk[:int((t_0 - w / 2) * Fs)] = 0
                    #     LL_t_pk[int((t_0 + WOI + 3 * w / 2) * Fs):] = 0
                    #     t_pk = sf.search_sequence_numpy(LL_t_pk, np.ones((int((w) * Fs),)))
                    #
                    #     # artefact: when LL during baseline is already quite high and there is no increase during "response", post stim period
                    #     LL_t_pk = np.array(LL_CC[i] >= 3 * thr) * 1
                    #     LL_t_pk[int((t_0) * Fs):] = 0
                    #     t_pk_art = sf.search_sequence_numpy(LL_t_pk, np.ones((int((w / 3 * 2) * Fs),)))
                    #     art_thr = np.mean(LL_CC[i, int((t_0 - 0.25) * Fs):int((t_0) * Fs)]) + 3 * np.std(
                    #         LL_CC[i, int((t_0 - 0.25) * Fs):int((t_0) * Fs)])
                    #     if (len(t_pk_art) > 0) & (LL_WOI < art_thr):
                    #         art = 1
                    #
                    #     sig_w = np.array((len(t_pk) > 0)) * 1

                    arr = np.array([[sc, rc, i, LL_WOI, WOI, LL_peak, sig+sig_mean, sig_BL, art, d]])
                    arr = pd.DataFrame(arr, columns=['Stim', 'Chan', 'CC', 'LL_WOI', 't_WOI', 'LL_pk', 'sig',
                                                     'sig_w', 'art', 'd'])
                    if start:
                        CC_summ = arr
                        start = 0
                    else:
                        CC_summ = pd.concat([CC_summ, arr])
                        CC_summ = CC_summ.reset_index(drop=True)
    return CC_summ


def get_sig_trial(sc, rc, con_trial, M_GT, t_resp, EEG_CR, test=1, p=90, exp=2, w_cluster=0.25, t_0=1, t_0_BL=0.5,
                  Fs=500):
    req = (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1)
    dat = con_trial[req].reset_index(drop=True)
    EEG_trials = ff.lp_filter(np.expand_dims(EEG_CR[rc, dat.Num.values.astype('int'), :], 0), 45, Fs)

    LL_trials = LLf.get_LL_all(EEG_trials, Fs, w_cluster)
    if test:
        # for each trial get significance level based on surrogate (Pearson^2 * LL)
        #### first get surrogate data
        pear_surr_all = []
        for t_test in [2, 2.5, 3]:  # surrogates times, todo: in future blockwise
            pear = np.zeros((len(EEG_trials[0]),)) - 1  # pearson to each CC
            for n_c in range(len(M_GT)):
                pear = np.max([pear, sf.get_pearson2mean(M_GT[n_c, :], EEG_trials[0], tx=t_0 + t_resp, ty=t_test,
                                                         win=w_cluster,
                                                         Fs=500)], 0)
            LL = LL_trials[0, :, int((t_test + w_cluster / 2) * Fs)]
            pear_surr = np.sign(pear) * abs(pear ** exp) * LL
            pear_surr_all = np.concatenate([pear_surr_all, pear_surr])

        # other surr trials
        real_trials = np.unique(
            con_trial.loc[req, 'Num'].values.astype('int'))
        # trials where RC is stimulating
        stim_trials = np.unique(
            con_trial.loc[(con_trial.Stim >= rc - 1) & (con_trial.Stim <= rc + 1), 'Num'].values.astype('int'))
        StimNum = np.random.choice(np.unique(con_trial.loc[~np.isin(con_trial.Stim, [sc-1, sc, sc+1])&~np.isin(con_trial.Stim, [rc-1, rc, rc+1])&(con_trial.Artefact==0),'Num']), size=200)
        StimNum = [i for i in StimNum if i not in stim_trials]
        StimNum = [i for i in StimNum if i not in stim_trials + 1]
        StimNum = [i for i in StimNum if i not in real_trials]

        StimNum = np.unique(StimNum).astype('int')
        EEG_surr = ff.lp_filter(np.expand_dims(EEG_CR[rc, StimNum, :], 0), 45, Fs)
        bad_StimNum = np.where(np.max(abs(EEG_surr[0]), 1) > 1000)
        if (len(bad_StimNum[0]) > 0):
            StimNum = np.delete(StimNum, bad_StimNum)
            EEG_surr = ff.lp_filter(np.expand_dims(EEG_CR[rc, StimNum, :], 0), 45, Fs)
        if len(StimNum)>0:
            LL_surr = LLf.get_LL_all(EEG_surr, Fs, w_cluster)
            for t_test in [2, 2.5, 3]:  # surrogates times, todo: in future blockwise
                pear = np.zeros((len(EEG_surr[0]),)) - 1
                for n_c in range(len(M_GT)):
                    pear = np.max([pear, sf.get_pearson2mean(M_GT[n_c, :], EEG_surr[0], tx=t_0 + t_resp, ty=t_test,
                                                             win=w_cluster,
                                                             Fs=500)], 0)

                LL = LL_surr[0, :, int((t_test + w_cluster / 2) * Fs)]
                # pear_surr = np.arctanh(np.max([pear,pear2],0))*LL
                pear_surr = np.sign(pear) * abs(pear ** exp) * LL
                pear_surr_all = np.concatenate([pear_surr_all, pear_surr])

        ##### real trials
        t_test = t_0 + t_resp  # timepoint that i tested is identical to WOI
        pear = np.zeros((len(EEG_trials[0]),)) - 1
        for n_c in range(len(M_GT)):
            pear = np.max(
                [pear, sf.get_pearson2mean(M_GT[n_c, :], EEG_trials[0], tx=t_0 + t_resp, ty=t_test, win=w_cluster,
                                           Fs=500)], 0)

        LL = LL_trials[0, :, int((t_test + w_cluster / 2) * Fs)]
        compound_metric = np.sign(pear) * abs(pear ** exp) * LL
        sig = (compound_metric > np.nanpercentile(pear_surr_all, p)) * 1

        con_trial.loc[
            req, 'Sig'] = sig  # * sig_mean
        con_trial.loc[
            req, 'LL_WOI'] = LL
        con_trial.loc[
            req, 'rho'] = pear

    ##### real trials, LL pre stim
    t_test = t_0_BL + t_resp  # 

    LL_pre = LL_trials[0, :, int((t_test + w_cluster / 2) * Fs)]
    con_trial.loc[req, 'LL_pre'] = LL_pre
    return con_trial

def get_sig_trial_surr(sc, rc, con_trial, M_GT, t_resp, EEG_CR, exp=2, w_cluster=0.25, t_0=1,
                  Fs=500):
    req = (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1)
    dat = con_trial[req].reset_index(drop=True)
    EEG_trials = ff.lp_filter(np.expand_dims(EEG_CR[rc, dat.Num.values.astype('int'), :], 0), 45, Fs)

    LL_trials = LLf.get_LL_all(EEG_trials, Fs, w_cluster)

    # for each trial get significance level based on surrogate (Pearson^2 * LL)
    #### first get surrogate data
    pear_surr_all = []
    for t_test in [0.5, 2.1]:  # surrogates times, todo: in future blockwise
        pear = np.zeros((len(EEG_trials[0]),)) - 1  # pearson to each CC
        for n_c in range(len(M_GT)):
            pear = np.max([pear, sf.get_pearson2mean(M_GT[n_c, :], EEG_trials[0], tx=t_0 + t_resp, ty=t_test,
                                                     win=w_cluster,
                                                     Fs=500)], 0)
        print(pear.shape)
        LL = LL_trials[0, :, int((t_test + w_cluster / 2) * Fs)]
        pear_surr = np.sign(pear) * abs(pear ** exp) * LL # square the correlation but keep the sign
        pear_surr_all = np.concatenate([pear_surr_all, pear_surr])

    # other surr trials
    real_trials = np.unique(
        con_trial.loc[req, 'Num'].values.astype('int'))
    # trials where RC is stimulating
    stim_trials = np.unique(
        con_trial.loc[(con_trial.Stim >= rc - 1) & (con_trial.Stim <= rc + 1), 'Num'].values.astype('int'))
    StimNum = np.random.choice(np.unique(con_trial.loc[~np.isin(con_trial.Stim, [sc-1, sc, sc+1])&~np.isin(con_trial.Stim, [rc-1, rc, rc+1])&(con_trial.Artefact==0),'Num']), size=200)
    StimNum = [i for i in StimNum if i not in stim_trials]
    StimNum = [i for i in StimNum if i not in stim_trials + 1]
    StimNum = [i for i in StimNum if i not in real_trials]

    StimNum = np.unique(StimNum).astype('int')
    EEG_surr = ff.lp_filter(np.expand_dims(EEG_CR[rc, StimNum, :], 0), 45, Fs)
    bad_StimNum = np.where(np.max(abs(EEG_surr[0]), 1) > 1000)
    if (len(bad_StimNum[0]) > 0):
        StimNum = np.delete(StimNum, bad_StimNum)
        EEG_surr = ff.lp_filter(np.expand_dims(EEG_CR[rc, StimNum, :], 0), 45, Fs)
    if len(StimNum)>0:
        LL_surr = LLf.get_LL_all(EEG_surr, Fs, w_cluster)
        f = 1
        for t_test in [0.5, 2.1]:  # surrogates times, todo: in future blockwise
            pear = np.zeros((len(EEG_surr[0]),)) - 1
            for n_c in range(len(M_GT)):
                pear = np.max([pear, sf.get_pearson2mean(M_GT[n_c, :], EEG_surr[0], tx=t_0 + t_resp, ty=t_test,
                                                         win=w_cluster,
                                                         Fs=500)], 0)

            LL = LL_surr[0, :, int((t_test + w_cluster / 2) * Fs)]
            # pear_surr = np.arctanh(np.max([pear,pear2],0))*LL
            pear_surr = np.sign(pear) * abs(pear ** exp) * LL
            pear_surr_all = np.concatenate([pear_surr_all, pear_surr])

    return pear_surr_all
