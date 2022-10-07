import os
import numpy as np
import ntpath
import pandas as pd
from glob import glob
import basic_func as bf
import scipy
from scipy import signal
import freq_funcs as ff
import LL_funcs as LLf
import matplotlib.pyplot as plt
import Cluster_func as Cf
import scipy.stats as stats


def get_GT(sc, rc, LL_CCEP, EEG_resp, Fs=500, t_0=1, w_cluster=0.25, n_cluster=2):
    Fs_rs = 100
    num_rs = int(2000 / Fs * Fs_rs)
    w = 0.25
    lists = LL_CCEP[(LL_CCEP['Artefact'] == 0) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
    stimNum_all = lists.Num.values.astype('int')
    M_GT = np.zeros((n_cluster + 1, 2000))
    r = 0
    t_resp = 0
    if len(stimNum_all) > 0:
        t_nan = np.where(np.isnan(np.mean(EEG_resp[rc, stimNum_all, :], 1)) * 1)# [0][0]
        if len(t_nan[0])>0:
            stimNum_all = np.delete(stimNum_all, t_nan[0])
        if len(stimNum_all) > 0:
            resp_all = ff.lp_filter(np.nanmean(bf.zscore_CCEP(EEG_resp[rc,stimNum_all, :]), 0), 40, Fs)
            LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp_all, 0), 0), Fs, w, t_0, 0)
            thr = np.percentile(np.concatenate([LL_resp[0, 0, int((w / 2) * Fs):int((t_0 - w / 2) * Fs)],
                                                LL_resp[0, 0, int((2) * Fs):int((4 - w / 2) * Fs)]]),
                                99)  # LL_resp[0, 0, int((t_0+0.5) * Fs):] = 0 * Fs):] = 0
            LL_t = np.array(LL_resp[0, 0, :int((t_0 + 0.5) * Fs)] > thr) * 1
            t_resp_all = search_sequence_numpy(LL_t, np.ones((int((w+0.07) * Fs),)))
            if len(t_resp_all) > 0:
                t_resp = t_resp_all[0] / Fs - t_0 + w / 2
                r = 1
                if (t_resp < 0.01) | (t_resp > 0.5):
                    t_resp = 0
            else:
                r = 0
                t_resp = 0

            # get overall mean and two clusters
            EEG_trial = ff.lp_filter(EEG_resp[rc, stimNum_all, :], 45, Fs)
            EEG_trial = signal.resample(EEG_trial, int(num_rs), axis=1)
            EEG_trial = stats.zscore(EEG_trial, axis=1)
            cc, y, dist, dist_cc = Cf.dba_cluster(
                EEG_trial[:, int((t_0 + t_resp) * Fs_rs):int((t_0 + t_resp + w_cluster) * Fs_rs)], n_cluster)

            M_GT[0, :] = ff.lp_filter(np.nanmean(EEG_resp[rc, stimNum_all, :], 0), 45, Fs)
            for c in range(n_cluster):
                M_GT[c + 1, :] = ff.lp_filter(np.nanmean(EEG_resp[rc, stimNum_all[y == c], :], 0), 45, Fs)

    return M_GT, [t_resp, r]


def get_N1peaks_mean(sc, rc, LL_CCEP, EEG_resp, Int=6, t_0=1, Fs=500):
    w = 0.25
    if "Int" in LL_CCEP:
        Int = np.min([Int, np.max(LL_CCEP.Int)])
        lists = LL_CCEP[
            (LL_CCEP['Artefact'] == 0) & (LL_CCEP['Int'] >= Int) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
    else:
        lists = LL_CCEP[(LL_CCEP['Artefact'] == 0) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
    stimNum_all = lists.Num.values.astype('int')
    if len(stimNum_all) > 0:
        resp_all = bf.zscore_CCEP(ff.lp_filter(np.mean(EEG_resp[rc, stimNum_all, :], 0), 40, Fs))
        LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp_all, 0), 0), Fs, w, t_0, 0)
        LL_t = np.array(LL_resp[0, 0, :] > np.percentile(LL_resp[0, 0, 0:int((t_0 - w / 2) * Fs)], 99)) * 1
        start_resp = search_sequence_numpy(LL_t, np.ones((int(0.1 * Fs),)))
        if len(start_resp) > 0:
            start_resp = start_resp[0] / Fs - t_0 + w / 2
            if start_resp < 0.01:
                start_resp = 0

            pk, pk_s, p = LLf.get_peaks_all(resp_all, start_resp)

        else:
            pk_s = np.zeros((4,))
            p = 0
            start_resp = 0
    else:
        pk_s = np.zeros((4,))
        p = np.nan
        start_resp = 0
    return pk_s[0], p, start_resp


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


def concat_resp_condition(subj, folder='InputOutput', cond_folder='CR'):
    path_patient_analysis = 'y:\\eLab\Projects\EL_experiment\Analysis\Patients\\' + subj
    files = glob(path_patient_analysis + '\\' + folder + '\\data\\Stim_list_*' + cond_folder + '*')
    files = np.sort(files)
    # prots           = np.int64(np.arange(1, len(files) + 1))  # 43
    stimlist = []
    EEG_resp = []
    conds = np.empty((len(files),), dtype=object)
    for p in range(len(files)):
        file = files[p]
        # file = glob(self.path_patient + '/Analysis/'+folder+'/data/Stim_list_' + str(p) + '_*')[0]
        idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
        cond = ntpath.basename(file)[idxs[-2] - 2:idxs[-2]]  # ntpath.basename(file)[idxs[-2] + 2:-4]  #
        conds[p] = cond
        EEG_block = np.load(path_patient_analysis + '\\' + folder + '\\data\\All_resps_' + file[-11:-4] + '.npy')
        print(str(p + 1) + '/' + str(len(files)) + ' -- All_resps_' + file[-11:-4])
        stim_table = pd.read_csv(file)
        stim_table['type'] = cond
        if len(stimlist) == 0:
            EEG_resp = EEG_block
            stimlist = stim_table
        else:
            EEG_resp = np.concatenate([EEG_resp, EEG_block], axis=1)
            stimlist = pd.concat([stimlist, stim_table])
    stimlist = stimlist.drop(columns="StimNum", errors='ignore')
    stimlist = stimlist.fillna(0)
    stimlist = stimlist.reset_index(drop=True)
    col_drop = ["StimNum", 'StimNum.1', 's', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
    for d in range(len(col_drop)):
        if (col_drop[d] in stimlist.columns):
            stimlist = stimlist.drop(columns=col_drop[d])
    stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)
    np.save(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy',
            EEG_resp)
    stimlist.to_csv(
        path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\stimlist_' + cond_folder + '.csv',
        index=False,
        header=True)  # scat_plot
    print('data stored')
    return EEG_resp, stimlist


def start_subj_GT(subj, folder='BrainMapping', cond_folder='CR',rerun=0):
    ## path_patient_analysis = 'T:\EL_experiment\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_patient_analysis = 'y:\eLab\Projects\EL_experiment\Analysis\Patients\\' + subj

    file_t_resp = path_patient_analysis + '\\' + folder + '\\data\\M_t_resp.npy'
    file_GT = path_patient_analysis + '\\' + folder + '\\data\\M_GT.npy'
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    con_trial = pd.read_csv(file_con)

    EEG_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy'
    #if not os.path.isfile(file_GT):
    if rerun:
        if not os.path.isfile(EEG_CR_file):
            EEG_resp, stimlist = concat_resp_condition(subj, folder=folder, cond_folder=cond_folder)

        else:
            EEG_resp = np.load(EEG_CR_file)

        chan_all = np.unique(con_trial.Chan)
        n_chan = np.max(chan_all).astype('int') + 1
        M_GT_all = np.zeros((n_chan, n_chan, 3, 2000))
        M_t_resp = np.zeros((n_chan, n_chan, 2))
        M_t_resp[:, :, 1] = -1
        for sc in np.unique(con_trial.Stim):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0)&(con_trial.Stim == sc), 'Chan']).astype('int')
            for rc in resp_chans:
                M_GT_all[sc, rc, :, :], M_t_resp[sc, rc, :] = get_GT(sc, rc, con_trial, EEG_resp)
        np.save(file_GT, M_GT_all)
        np.save(file_t_resp, M_t_resp)
        print(subj + ' -- DONE --')
    else:
        print(file_GT + ' -- already exists')


def start_subj(subj, folder='BrainMapping', cond_folder='CR'):
    path_patient_analysis = 'T:\EL_experiment\Projects\EL_experiment\Analysis\Patients\\' + subj
    file_MN1 = path_patient_analysis + '\\' + folder + '\\data\\M_N1.npy'
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    con_trial = pd.read_csv(file_con)

    EEG_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy'
    # if not os.path.isfile(file_MN1):
    # if os.path.isfile(EEG_CR_file):
    EEG_resp, stimlist = concat_resp_condition(subj, folder=folder, cond_folder=cond_folder)
    reload = 0
    # else:
    #     reload = 1 #EEG_resp = np.load(EEG_CR_file)
    # if not os.path.isfile(file_MN1):
    #     if reload:
    #         EEG_resp = np.load(EEG_CR_file)
    StimChanIx = np.unique(con_trial.Stim)
    chan_all = np.unique(con_trial.Chan)
    n_chan = np.max(chan_all).astype('int') + 1
    M_N1peaks = np.zeros((n_chan, n_chan, 3))
    for s in range(len(StimChanIx)):
        sc = StimChanIx[s].astype('int')
        for rc in range(n_chan):
            M_N1peaks[sc, rc, :] = get_N1peaks_mean(sc, rc, con_trial, EEG_resp)
    np.save(file_MN1, M_N1peaks)
    print(subj + ' -- DONE --')


for subj in ["EL011","EL015","EL016", "EL014", "EL012", "EL010",'EL013']: # "El014",
#     for f in ['BrainMapping', 'InputOutput']:
    start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', rerun=1)
# start_subj('EL004', folder='InputOutput', cond_folder='CR')
# start_subj('EL005', folder='InputOutput', cond_folder='CR')
# print('DONE')
