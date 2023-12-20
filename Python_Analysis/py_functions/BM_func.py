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
import CCEP_func

# if platform.system() == 'Windows':
#     regions = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\elab_labels.xlsx", sheet_name='regions', header=0)
#
# else:  # 'Darwin' for MAC
#     regions = pd.read_excel("/Volumes/EvM_T7/PhD/EL_experiment/Patients/all/elab_labels.xlsx", sheet_name='regions',
#                             header=0)
#
# color_regions = regions.color.values
# regions_G = regions.subregion.values
# regions = regions.label.values

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]


def init_stimlist_columns(stimlist, StimChanSM):
    """Initialize required columns if they are not present."""
    column_defaults = {"Num_block": stimlist.StimNum, "condition": 0, "sleep": 0}
    for col, default_val in column_defaults.items():
        if col not in stimlist.columns:
            stimlist[col] = default_val
    # Filter stimlist based on conditions
    stim_spec = stimlist[(stimlist.IPI_ms == 0) & (np.isin(stimlist.ChanP, StimChanSM)) & (stimlist.noise == 0)]
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
    #     rec_chan = bf.SM2IX(rec_chan, StimChanSM, np.array(StimChanIx))
    #     if np.isin(c, rec_chan):
    #         return 1 # recording channel has artefact because it's a recovering channel (stimulating just before)
    return bf.check_inStimChan(c, ChanP1, labels_clinic)  # stim channel is recording channel


def LL_BM_connection(EEG_resp, stimlist, bad_chans, coord_all, labels_clinic, StimChanSM, StimChanIx):
    """Caulculate for each possible conenction the LL and keeps important factors (Time of Stim, distance, etc.)"""
    Fs = 500
    w_LL = 0.25
    t_0 = 1  # time of stimulation in data

    # Init required columns in stimlist
    stimlist, stim_spec = init_stimlist_columns(stimlist, StimChanSM)

    # Analyze each channel
    data_CCEP = []
    stimNum = stim_spec.StimNum.values  # [:,0]
    noise_val = stim_spec.noise.values  # [:,0]
    stimNum_block = stim_spec.Num_block.values  # [:,0]
    resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
    ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))

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
    pN = np.min(resps[:, :, 0:int(t_0 * Fs)], 2)
    pP = np.max(resps[:, :, 0:int(t_0 * Fs)], 2)
    p2p_BL = abs(pP - pN)
    for c in range(LL_peak.shape[0]):
        val = [
            [c, ChanP1[i], noise_val[i], stimNum_block[i], stim_spec.condition.values[i], stim_spec.date.values[i],
             stim_spec.sleep.values[i], stim_spec.stim_block.values[i], LL_peak[c, i], stim_spec.h.values[i],
             stimNum[i], p2p[c, i], p2p_BL[c, i]]
            for i in range(LL_peak.shape[1])
        ]
        val = np.array(val)
        # # Apply artefact logic
        # for v in val:
        #     v[2] = calculate_artefact(resps, stimlist, stim_spec, t_0, Fs, c, ChanP1, StimChanSM, StimChanIx,
        #                               labels_clinic)
        chan_stimulating = bf.check_inStimChan(c, ChanP1, labels_clinic)
        if len(chan_stimulating) > 0:
            indices = np.where(chan_stimulating == 1)[0]
            val[indices, 2] = 1

        # Convert the numpy array back to a list
        val = val.tolist()
        data_CCEP.extend(val)

    # Convert to DataFrame
    LL_CCEP = pd.DataFrame(data_CCEP, columns=["Chan", "Stim", "Artefact", "Num_block", "Condition", "Date", "Sleep",
                                               "Block", "LL", "Hour", "Num", "P2P", "P2P_BL"])

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


def cal_correlation_condition(con_trial, metric='LL', condition='Block'):
    """Create correlation matrix between the BM during different conditions (Block, Hour, Sleep) based on conenction strength metric (LL, P)"""
    # Clean table
    con_trial_cleaned = con_trial[con_trial.Artefact < 1].copy()
    con_trial_cleaned.loc[con_trial_cleaned.Sig < 0, 'Sig'] = np.nan

    # Calculate mean for significant trials
    con_trial_cleaned['m_sig'] = np.nan
    con_trial_cleaned.loc[con_trial_cleaned.Sig == 1, 'm_sig'] = con_trial_cleaned.loc[
        con_trial_cleaned.Sig == 1, metric]
    con_trial_cleaned['Prob'] = con_trial_cleaned.Sig

    # Create a pivot table
    con_pivot = con_trial_cleaned.pivot_table(index=['Stim', 'Chan'], columns=condition, values=['m_sig', 'Prob'],
                                              aggfunc='mean')

    # Fill missing values with global mean
    con_pivot_filled = con_pivot.fillna(con_trial_cleaned['m_sig'].mean())

    V = con_pivot_filled['m_sig'].values

    # Calculate the Pearson correlation matrix
    correlation_matrix = np.corrcoef(V, rowvar=False)

    return correlation_matrix, np.unique(con_trial_cleaned[condition])


def get_con_summary_SS(con_trial, con_summary_gen, CC_summ, EEG_resp, sleep='Wake', delay = 0):
    """Create summary table of each conenction showing mean response strength, probability, DI, distance and delay"""
    # Clean table
    con_trial_cleaned = con_trial[
        (con_trial.SleepState == sleep) & (con_trial.Sig > -1) & (con_trial.Artefact < 1)
        ].copy()

    # Calculate weighted LL (LL when non-significant trials are set to 0)
    con_trial_cleaned['LL_w'] = con_trial_cleaned['Sig'] * con_trial_cleaned['LL']

    # Define a custom aggregation dictionary
    aggregation = {
        'LL': ['mean', 'var'],  # Mean and variance of LL
        'LL_w': 'mean',  # Mean of weighted LL (LL when non-significant trials are set to 0)
        'Sig': 'mean',  # Mean significance
        'd': 'mean'  # Mean significance
    }

    # Calculate metrics for all trials
    con_summary = con_trial_cleaned.groupby(['Stim', 'Chan']).agg(aggregation)

    # Flatten MultiIndex columns
    con_summary.columns = ['_'.join(col).strip() for col in con_summary.columns.values]

    # Calculate metrics for significant trials only
    sig_trials = con_trial_cleaned[con_trial_cleaned.Sig == 1]
    sig_metrics = sig_trials.groupby(['Stim', 'Chan'])['LL'].agg(['mean', 'var'])

    # Rename columns for clarity
    sig_metrics = sig_metrics.rename(columns={'mean': 'LL_sig_mean', 'var': 'LL_sig_var'})

    # Merge the two summaries
    final_summary = pd.merge(con_summary, sig_metrics, on=['Stim', 'Chan'], how='left')
    final_summary = final_summary.rename(columns={'d_mean': 'd', 'Sig_mean': 'Sig'})

    # con_summary = con_summary[(con_summary.Sig >0)]
    # CC_summ = CC_summ[(CC_summ.sig == 1)]
    # CC_summ = CC_summ.groupby(['Stim', 'Chan'], as_index=False)[['t_WOI']].mean()
    final_summary = pd.merge(final_summary, con_summary_gen[['Stim', 'Chan', 't_WOI', 'delay']], on=['Stim', 'Chan'], how='outer')
    # con_summary.insert(4, 'DI', np.nan)  # asym[rc, sc, 1]
    ## adding DI value
    # Ensure that for every A->B, B->A also exists in the dataframe
    df = final_summary.copy()
    all_pairs = pd.concat([df[['Stim', 'Chan', 'Sig']],
                           df.rename(columns={'Stim': 'Chan', 'Chan': 'Stim'})[['Chan', 'Stim', 'Sig']]]).reset_index(
        drop=True)
    # all_pairs.loc[np.isnan(all_pairs.Sig), 'Sig'] = 0
    # Group by both nodes and calculate max and min Sig for each group, and get 'is_min' for each row in 'all_pairs'
    grouped = all_pairs.groupby(['Stim', 'Chan'], as_index=False)['Sig'].agg(['max', 'min']).reset_index()
    all_pairs = all_pairs.merge(grouped, on=['Stim', 'Chan'])
    all_pairs['is_min'] = (all_pairs['Sig'] == all_pairs['min']) & (all_pairs['Sig'] != all_pairs['max'])

    # Calculate the DI column values
    all_pairs['DI'] = np.where((all_pairs['min'] == 0) & (all_pairs['max'] == 0), np.nan,
                               np.where(all_pairs['min'] == all_pairs['max'], 0,
                                        abs(all_pairs['min'] - all_pairs['max'])*(1 - all_pairs['min'] / all_pairs['max']) * all_pairs['is_min'].map(
                                            {True: -1, False: 1})))
    # Merge the DI and 'is_min' columns back to the original dataframe
    con_summary = df.merge(all_pairs[['Stim', 'Chan', 'DI', 'Sig']],
                           on=['Stim', 'Chan', 'Sig'],
                           how='left')
    con_summary.insert(2, 'SleepState', sleep)
    con_summary = con_summary.drop_duplicates().reset_index(drop=True)
    if delay:
        con_summary.delay = np.nan
        con_summary['Num_sig_trial'] = 0
        con_summary['Mean_P2P'] = np.nan
        con_summary['Mean_LL'] = np.nan
        con_summary = get_delay(con_summary, CC_summ, con_trial_cleaned, EEG_resp)
    return con_summary


def get_con_summary(con_trial, CC_summ, EEG_resp):
    """Create summary table of each conenction showing mean response strength, probability, DI, distance and delay"""
    # Clean table
    con_trial_cleaned = con_trial[(con_trial.Sig > -1) & (con_trial.Artefact < 1)].copy()
    con_trial_cleaned['LL_sig'] = con_trial_cleaned['Sig'] * con_trial_cleaned['LL']
    con_summary = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)[['LL', 'LL_sig', 'd',
                                                                               'Sig']].mean()  # con_summary = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)['LL', 'LL_sig', 'd', 'Sig'].mean()
    # CC_summ = CC_summ[(CC_summ.sig == 1)]
    CC_summ = CC_summ.groupby(['Stim', 'Chan'], as_index=False)[['t_WOI']].mean()
    con_summary = pd.merge(con_summary, CC_summ, on=['Stim', 'Chan'], how='outer')
    # con_summary.insert(4, 'DI', np.nan)  # asym[rc, sc, 1]
    ## adding DI value
    # Ensure that for every A->B, B->A also exists in the dataframe
    df = con_summary.copy()
    all_pairs = pd.concat([df[['Stim', 'Chan', 'Sig']],
                           df.rename(columns={'Stim': 'Chan', 'Chan': 'Stim'})[['Chan', 'Stim', 'Sig']]]).reset_index(
        drop=True)

    # Group by both nodes and calculate max and min Sig for each group, and get 'is_min' for each row in 'all_pairs'
    grouped = all_pairs.groupby(['Stim', 'Chan'], as_index=False)['Sig'].agg(['max', 'min']).reset_index()
    all_pairs = all_pairs.merge(grouped, on=['Stim', 'Chan'])
    all_pairs['is_min'] = (all_pairs['Sig'] == all_pairs['min']) & (all_pairs['Sig'] != all_pairs['max'])

    # Calculate the DI column values
    # Calculate the DI column values
    all_pairs['DI'] = np.where((all_pairs['min'] == 0) & (all_pairs['max'] == 0), np.nan,
                               np.where(all_pairs['min'] == all_pairs['max'], 0,
                                        abs(all_pairs['min'] - all_pairs['max'])*(1 - all_pairs['min'] / all_pairs['max']) * all_pairs['is_min'].map(
                                            {True: -1, False: 1})))
    # Merge the DI and 'is_min' columns back to the original dataframe
    con_summary = df.merge(all_pairs[['Stim', 'Chan', 'DI', 'Sig']],
                           on=['Stim', 'Chan', 'Sig'],
                           how='left')
    con_summary = con_summary.drop_duplicates().reset_index(drop=True)
    con_summary = get_delay(con_summary, CC_summ, con_trial_cleaned, EEG_resp)

    return con_summary
def get_delay(con_summary, CC_summ, con_trial_cleaned, EEG_resp, t0 = 1, Fs = 500):
    # add delay
    for i in range(len(con_summary[con_summary.Sig > 0])):
        sc = con_summary.loc[con_summary.Sig > 0, 'Stim'].values[i]
        rc = con_summary.loc[con_summary.Sig > 0, 'Chan'].values[i]
        if (sc==6) & (rc==64):
            print('stop')
        # only significant trials
        num = con_trial_cleaned.loc[(con_trial_cleaned.Stim == sc) & (con_trial_cleaned.Chan == rc) & (
                con_trial_cleaned.Sig == 1), 'Num'].values
        # only significant trials
        num_z = con_trial_cleaned.loc[(con_trial_cleaned.Stim == sc) & (con_trial_cleaned.Chan == rc), 'Num'].values
        WOI = CC_summ.loc[(CC_summ.Stim == sc) & (CC_summ.Chan == rc), 't_WOI'].values
        if len(WOI) > 0:
            WOI = WOI[0]
        else:
            WOI = 0


        trials_score = ff.bp_filter(EEG_resp[rc, num, :],2 , 45, Fs) # EEG_resp[rc, num, :] ##CCEP_func.zscore_CCEP(EEG_resp[rc, num, :], t_0=1, w0=0.2, Fs=500)
        signal = np.nanmean(trials_score, 0)
        P2P = np.max(signal[int(1*Fs):int((t0+WOI+0.25)*Fs)])- np.min(signal[int(t0*Fs):int((t0+WOI+0.25)*Fs)])
        LL_transform = LLf.get_LL_all(np.expand_dims(signal, [0, 1]), Fs, 0.25)[0, 0]
        LL_pk = np.max(LL_transform[int((t0-0.125)*Fs):int((t0+WOI+0.25)*Fs)])
        # signal = np.nanmean(EEG_resp[rc, num, :], 0)
        # delay, _, _, _ = CCEP_func.cal_delay(signal, WOI=WOI)
        delay = CCEP_func.CCEP_onset(signal, WOI=WOI)
        zscore = CCEP_func.zscore_CCEP(np.nanmean(EEG_resp[rc, num_z, :],0), t_0=t0, w0=0.2,w1 = 0.05,  Fs=500)
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'delay'] = delay
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Num_sig_trial'] = len(num)
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Mean_P2P'] = P2P
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Mean_LL'] = LL_pk
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Zscore'] = np.max(abs(zscore[int(Fs*(t0+0.05)):int(Fs*(t0+0.5))]))
    return con_summary
def get_peaks_AUC(data, EEG_resp, t0=1, Fs=500):
    new_lab = ['N1', 'N2', 'AUC', 'P2P_1s']
    for l in new_lab:
        if l not in data:
            data.insert(6, l, np.nan)
    for sc in np.unique(data.Stim).astype('int'):
        for rc in np.unique(data.loc[(data.Artefact <1)&(data.Stim == sc), 'Chan']).astype('int'):
            lists = data[(data['Chan'] == rc) & (data['Stim'] == sc)].reset_index(drop=True)
            stimNum_all = lists.Num.values.astype('int')
            N1, N2, AUC, P2P = CCEP_func.CCEP_metric(EEG_resp[rc, stimNum_all, :], t0=t0, w_AUC=1, Fs=Fs)
            data.loc[(data['Chan'] == rc) & (data['Stim'] == sc), 'N1'] = N1
            data.loc[(data['Chan'] == rc) & (data['Stim'] == sc), 'N2'] = N2
            data.loc[(data['Chan'] == rc) & (data['Stim'] == sc), 'AUC'] = AUC
            data.loc[(data['Chan'] == rc) & (data['Stim'] == sc), 'P2P_1s'] = P2P


    return data