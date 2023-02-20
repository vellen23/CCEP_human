import os
import numpy as np
import pandas as pd
import basic_func as bf
import freq_funcs as ff
import LL_funcs as LLf
import Cluster_func as Cf
from scipy.spatial import distance

import significance_funcs as sf

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def get_GT_trials(trials, real=1, t_0=1, Fs=500, w=0.25, n_cluster=2):
    # t_0 stimulation time in epoched data, changed in surrogate data
    # trials = EEG_resp[rc, stimNum_all, :], shape (n,2000)
    # 1. z-score trials and get LL for t_onset and t_max
    EEG_trial = ff.bp_filter(trials, 1, 40, Fs)
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
    EEG_trial = bf.zscore_CCEP(EEG_trial, t_0, Fs)
    cc, y = Cf.ts_cluster(
        EEG_trial[:, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + w) * Fs)], n=n_cluster,
        method='euclidean')
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
    return [r, t_onset, t_WOI], LL_mean[0, 0], p_CC, M_GT, y, thr, LL_CC



def get_GT(sc, rc, LL_CCEP, EEG_resp, Fs=500, t_0=1, w_cluster=0.25, n_cluster=2):
    # for each connection (A-B) the two CC are calculated
    lists = LL_CCEP[(LL_CCEP['Artefact'] < 1) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
    stimNum_all = lists.Num.values.astype('int')
    M_GT = np.zeros((n_cluster + 1, 2000))
    r = -1
    t_onset = -1
    t_WOI = -1
    p_CC = -1
    LL_CC = np.zeros((1, 2))
    if len(stimNum_all) > 0:
        t_nan = np.where(np.isnan(np.mean(EEG_resp[rc, stimNum_all, :], 1)) * 1)  # [0][0]
        if len(t_nan[0]) > 0:
            stimNum_all = np.delete(stimNum_all, t_nan[0])
        if len(stimNum_all) > 15:
            trials = EEG_resp[rc, stimNum_all, :]
            [r, t_onset, t_WOI], _, _, M_GT, y, _, LL_CC = get_GT_trials(trials, real=1, t_0=t_0, Fs=Fs,
                                                                         w=w_cluster,
                                                                         n_cluster=n_cluster)
    return M_GT, [r, t_onset, t_WOI], LL_CC


def get_CC_surr(rc, LL_CCEP, EEG_resp, n_trials, Fs=500, w_cluster=0.25, n_cluster=2, n_surr=200):
    # general infos
    # non-stim surrogate trials
    # other trials
    stim_trials = np.unique(
        LL_CCEP.loc[(LL_CCEP.Stim >= rc - 1) & (LL_CCEP.Stim <= rc + 1), 'Num'].values.astype('int'))
    StimNum = np.unique(LL_CCEP.loc[(LL_CCEP.d > 11) & (LL_CCEP.Artefact == 0) & (
            LL_CCEP.Chan == rc), 'Num'])  # np.random.choice(np.unique(con_trial.Num), size=400)
    StimNum = [i for i in StimNum if i not in stim_trials]
    StimNum = [i for i in StimNum if i not in stim_trials + 1]
    StimNum = [i for i in StimNum if i not in stim_trials - 1]

    # initiate
    # pear_surr = np.zeros((1,)) - 1
    # epoch: [-1,3]s: switch [-1,1] and [1,3] to have non stimulating data after onset at 0
    LL_surr = np.zeros((n_surr, 2))
    WOI_surr = np.zeros((n_surr,))
    LL_surr_data = np.zeros((n_surr, 2, 2000))
    for i in range(n_surr):
        StimNum_surr = np.random.choice(StimNum, size=n_trials).astype('int')
        t_nan = np.where(np.isnan(np.mean(EEG_resp[rc, StimNum_surr, :], 1)) * 1)  # [0][0]
        if len(t_nan[0]) > 0:
            StimNum_surr = np.delete(StimNum_surr, t_nan[0])
        # get overall mean and two clusters
        trials = EEG_resp[rc, StimNum_surr, :]
        trials[:, 0:1000] = EEG_resp[rc, StimNum_surr, 1000:]  # put
        trials[:, 1000:] = np.flip(EEG_resp[rc, StimNum_surr, 1000:], 1)  # put response in the beginning of the epoch
        for t_0 in [1]:
            [r, t_onset, t_WOI], _, p_CC, M_GT, y, _, LL_CC = get_GT_trials(trials, real=0, t_0=t_0, Fs=Fs, w=w_cluster,
                                                                            n_cluster=n_cluster)
            # return [r, t_onset, t_WOI], LL_mean[0, 0], p_CC, M_GT, y, thr, LL_CC
            if (sum(y == 0) > np.max([5, 0.05 * len(y)])) & (sum(y == 1) > np.max([5, 0.05 * len(y)])):
                # pear_surr = np.concatenate([pear_surr, [p_CC]], 0)
                LL_surr[i] = LL_CC
                WOI_surr[i] = t_WOI
                LL_surr_data[i] = M_GT[1:, :]
            else:
                LL_surr[i, :] = np.nan
                WOI_surr[i] = np.nan
                LL_surr_data[i, 0, :] = np.nan
                LL_surr_data[i, 1, :] = np.nan
    return LL_surr, LL_surr_data, WOI_surr
    # return pear_surr, [np.percentile(pear_surr, 95), np.percentile(pear_surr, 99)], LL_surr[1:, :]


def get_CC_summ(M_GT_all, M_t_resp, surr_thr, coord_all, t_0=1, w=0.25, Fs=500):
    # creates a table for each stim-chan pair with the two CC found indicating the WOI, LL and whether it's signficant
    start = 1
    for sc in range(M_GT_all.shape[0]):
        for rc in range(M_GT_all.shape[0]):
            d = np.round(distance.euclidean(coord_all[sc], coord_all[rc]), 2)
            LL_CC = LLf.get_LL_all(np.expand_dims(M_GT_all[sc, rc, :], 0), Fs, 0.25)[0]
            WOI = M_t_resp[sc, rc, 2]
            if M_t_resp[sc, rc, 0] > -1:
                thr = surr_thr.loc[surr_thr.Chan == rc, 'CC_LL95'].values[0]
                thr50 = surr_thr.loc[surr_thr.Chan == rc, 'CC_LL50'].values[0]
                # thr50 = thr / 2  # todo: NEW way to get onset time (2nd derivative) and define artefacts
                for i in range(1, 3):
                    t_onset = np.nan
                    LL_WOI = LL_CC[i, int((t_0 + WOI + w / 2) * Fs)]
                    #### Response onset: when LL surpasses 50th percentile for at least 250ms
                    LL_t = np.array(LL_CC[i] >= thr50) * 1
                    LL_t[:int((t_0 - w / 2) * Fs)] = 0
                    LL_t[int((t_0 + 0.4 + w / 2) * Fs):] = 0

                    t_resp_all = sf.search_sequence_numpy(LL_t, np.ones((int((3 / 2 * w) * Fs),)))
                    if len(t_resp_all) > 0:
                        t_onset = t_resp_all[0] / Fs - t_0 + w / 2
                        if t_onset < 0: t_onset = 0

                    #### Check whether LL surpasses the threshold for some time
                    LL_t_pk = np.array(LL_CC[i] >= thr) * 1
                    LL_t_pk[:int((t_0 - w / 2) * Fs)] = 0
                    LL_t_pk[int((t_0 + 0.4 + w / 2) * Fs):] = 0
                    t_pk = sf.search_sequence_numpy(LL_t_pk, np.ones((int((w) * Fs),)))

                    LL_peak = np.max(LL_CC[i, int((t_0 + w / 2) * Fs):int((t_0 + 0.5 - w / 2) * Fs)])

                    ### artefact:
                    LL_t_pk = np.array(LL_CC[i] >= thr) * 1
                    LL_t_pk[int((t_0) * Fs):] = 0
                    t_art = sf.search_sequence_numpy(LL_t_pk, np.ones((int((w) * Fs),)))

                    sig = np.array(LL_WOI > thr) * 1
                    sig_w = np.array((len(t_pk) > 0)) * 1
                    art = np.array((len(t_art) > 0)) * 1
                    arr = np.array([[sc, rc, i, LL_WOI, WOI, LL_peak, t_onset, sig, sig_w, art, d]])
                    arr = pd.DataFrame(arr, columns=['Stim', 'Chan', 'CC', 'LL_WOI', 't_WOI', 'LL_pk', 'onset', 'sig',
                                                     'sig_w', 'art', 'd'])
                    if start:
                        CC_summ = arr
                        start = 0
                    else:
                        CC_summ = pd.concat([CC_summ, arr])
                        CC_summ = CC_summ.reset_index(drop=True)
    return CC_summ


def get_sig_trial(sc, rc, con_trial, M_GT, t_resp, EEG_CR, test = 1, p=90, exp=2, w_cluster=0.25, t_0=1, t_0_BL=0.48, Fs=500):
    dat = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1)]
    EEG_trials = ff.lp_filter(EEG_CR[[[rc]], dat.Num.values.astype('int'), :], 45, Fs)
    LL_trials = LLf.get_LL_all(EEG_trials, Fs, w_cluster)
    if test:
        # for each trial get significance level based on surrogate (Pearson^2 * LL)
        #### first get surrogate data
        pear_surr_all = []
        for t_test in [0.3, 0.7, 1.8, 2.2, 2.6]:  # surrogates times, todo: in future blockwise
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
            con_trial.loc[(con_trial.Stim == sc) & (con_trial.Chan == rc), 'Num'].values.astype('int'))
        stim_trials = np.unique(
            con_trial.loc[(con_trial.Stim >= rc - 1) & (con_trial.Stim <= rc + 1), 'Num'].values.astype('int'))
        StimNum = np.random.choice(np.unique(con_trial.Num), size=400)
        StimNum = [i for i in StimNum if i not in stim_trials]
        StimNum = [i for i in StimNum if i not in stim_trials + 1]
        StimNum = [i for i in StimNum if i not in real_trials]

        StimNum = np.unique(StimNum).astype('int')
        EEG_surr = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 45, Fs)
        bad_StimNum = np.where(np.max(abs(EEG_surr[0]), 1) > 1000)
        if (len(bad_StimNum[0]) > 0):
            StimNum = np.delete(StimNum, bad_StimNum)
            EEG_surr = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 45, Fs)
        LL_surr = LLf.get_LL_all(EEG_surr, Fs, w_cluster)
        f = 1
        for t_test in [0.3, 0.7, 1.8, 2.2, 2.6]:  # surrogates times, todo: in future blockwise
            s = (-1) ** f
            pear = np.zeros((len(EEG_surr[0]),)) - 1
            for n_c in range(len(M_GT)):
                pear = np.max([pear, sf.get_pearson2mean(M_GT[n_c, :], s * EEG_surr[0], tx=t_0 + t_resp, ty=t_test,
                                                         win=w_cluster,
                                                         Fs=500)], 0)

            LL = LL_surr[0, :, int((t_test + w_cluster / 2) * Fs)]
            # pear_surr = np.arctanh(np.max([pear,pear2],0))*LL
            pear_surr = np.sign(pear) * abs(pear ** exp) * LL
            pear_surr_all = np.concatenate([pear_surr_all, pear_surr])
            f = f + 1

        ##### real trials
        t_test = t_0 + t_resp  # timepoint that i tested is identical to WOI
        pear = np.zeros((len(EEG_trials[0]),)) - 1
        for n_c in range(len(M_GT)):
            pear = np.max(
                [pear, sf.get_pearson2mean(M_GT[n_c, :], EEG_trials[0], tx=t_0 + t_resp, ty=t_test, win=w_cluster,
                                           Fs=500)], 0)

        LL = LL_trials[0, :, int((t_test + w_cluster / 2) * Fs)]
        pear = np.sign(pear) * abs(pear ** exp) * LL
        sig = (pear > np.nanpercentile(pear_surr_all, p)) * 1


        con_trial.loc[
            (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1), 'Sig'] = sig  # * sig_mean
        con_trial.loc[
            (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1), 'LL_WOI'] = LL

    ##### real trials, LL pre stim
    t_test = t_0_BL + t_resp  # timepoint that i tested is identical to WOI

    LL_pre = LL_trials[0, :, int((t_test + w_cluster / 2) * Fs)]
    con_trial.loc[
        (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1), 'LL_pre'] = LL_pre
    return con_trial

# def start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', load_con=1, load_surr=1, surr_plot=1):
#     print(subj + ' -- START --')
#     ## path_patient_analysis = 'T:\EL_experiment\Projects\EL_experiment\Analysis\Patients\\' + subj
#     path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
#     path_gen = os.path.join(sub_path, 'Patients', subj)
#     if not os.path.exists(path_gen):
#         print("Can't find path")
#         # path_gen = 'T:\\EL_experiment\\Patients\\' + subj
#         return
#     path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
#     path_infos = os.path.join(path_patient, 'infos')
#     if not os.path.exists(path_infos):
#         path_infos = path_gen + '\\infos'
#
#     lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
#     labels_all = lbls.label.values
#
#     # files of interests
#     file_t_resp = path_patient_analysis + '\\' + folder + '\\data\\M_tresp.npy'  # for each connection: LLsig (old), t_onset (old), t_resp, CC_p, CC_LL1, CC_LL2
#     file_CC_surr = path_patient_analysis + '\\' + folder + '\\data\\M_CC_surr.csv'
#     file_CC_LL_surr = path_patient_analysis + '\\' + folder + '\\data\\LL_CC_surr.npz'
#     file_GT = path_patient_analysis + '\\' + folder + '\\data\\M_CC.npy'
#     file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
#     file_CC_summ = path_patient_analysis + '\\' + folder + '\\data\\CC_summ.csv'
#
#     # load required data
#     con_trial = pd.read_csv(file_con)
#     con_trial.loc[np.isnan(con_trial.P2P), 'Artefact'] = 1
#     con_trial.loc[con_trial.LL == 0, 'Artefact'] = 1
#     EEG_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy'
#     # if not os.path.isfile(file_GT):
#     ##### 1. get cluster centers and t_resp, t_onset for each possible connection
#     EEG_resp = []
#     if os.path.isfile(file_GT) * load_con:
#         # print(file_GT + ' -- already exists')
#         M_t_resp = np.load(file_t_resp)
#         # M_GT_all = np.load(file_GT)
#     else:
#         if not os.path.isfile(EEG_CR_file):
#             print('Run concat.py first to get concatenated EEG data')
#             return
#         else:
#             print('loading EEG data ..... ')
#             EEG_resp = np.load(EEG_CR_file)
#
#         chan_all = np.unique(con_trial.Chan)
#         n_chan = np.max(chan_all).astype('int') + 1
#         M_GT_all = np.zeros((n_chan, n_chan, 3, 2000))  # mean and 3 Cluster Centers
#         M_t_resp = np.zeros((n_chan, n_chan, 7))  # LL_sig (r), t_onset, t_WOI, pearson of GT (p_CC)
#         M_t_resp[:, :, 0] = -1  # sig_LL of mean
#         M_t_resp[:, :, 4] = -1  # sig_CCp
#         for sc in tqdm.tqdm(np.unique(con_trial.Stim)):
#             sc = int(sc)
#             resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0) & (con_trial.Stim == sc), 'Chan']).astype(
#                 'int')
#             for rc in resp_chans:
#                 # GT output: [r, t_onset, t_WOI, p_CC]
#                 M_GT_all[sc, rc, :, :], M_t_resp[sc, rc, :4], con_trial, M_t_resp[sc, rc, 5:] = get_GT(sc, rc,
#                                                                                                        con_trial,
#                                                                                                        EEG_resp)
#         np.save(file_GT, M_GT_all)
#         np.save(file_t_resp, M_t_resp)
#         con_trial.to_csv(file_con,
#                          index=False,
#                          header=True)
#         print(subj + ' -- DONE --')
#     fig_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\methods\\CC_surr_hist\\'
#
#     Path(fig_path).mkdir(
#         parents=True, exist_ok=True)
#     ###### 2. calculate for each recording channel surrogate CC
#     n_surr = 200
#     if os.path.isfile(file_CC_surr) * load_surr:
#         M_t_resp = np.load(file_t_resp)  ## LL_sig, t_onset, t_resp, pearson of GT
#         surr_thr = pd.read_csv(file_CC_surr)
#         update_sig_con = 0
#     else:
#         # M_CCp_surr = np.zeros((len(labels_all), 2))
#         M_CC_LL_surr = np.zeros((len(labels_all), 3))
#         CC_LL_surr = np.zeros((len(labels_all), n_surr, 2, 2000))
#         CC_WOI = np.zeros((len(labels_all), n_surr))
#         stims = np.zeros((len(labels_all), 1))
#         stims[:, 0] = np.arange(len(labels_all))
#
#         con_trial_n = con_trial[(con_trial.Artefact < 1) & (con_trial.LL > 0)]
#         summ = con_trial_n.groupby(['Stim', 'Chan'], as_index=False)[['LL']].count()
#         if len(EEG_resp) == 0:
#             print('loading EEG data ..... ')
#             EEG_resp = np.load(EEG_CR_file)
#         resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0), 'Chan']).astype(
#             'int')
#         for rc in tqdm.tqdm(resp_chans):
#             n_trials = np.median(summ.loc[summ.Chan == rc, 'LL']).astype('int')
#             LL_surr, CC_LL_surr[rc], CC_WOI[rc] = get_CC_surr(rc, con_trial, EEG_resp,
#                                                   n_trials)  # return LL_surr[1:, :], LL_surr_data[1:, :, :]
#             # pear_surr, M_CCp_surr[rc, :], LL_surr = get_CC_surr(rc, con_trial, EEG_resp, n_trials)
#             M_CC_LL_surr[rc, :] = [np.nanpercentile(LL_surr, 50), np.nanpercentile(LL_surr, 95),
#                                    np.nanpercentile(LL_surr, 99)]
#             ## pearson's surr test
#             # M_t_resp[M_t_resp[:, rc, 3] >= M_CCp_surr[rc, 1], rc, 4] = 1
#             # M_t_resp[M_t_resp[:, rc, 3] < M_CCp_surr[rc, 1], rc, 4] = 0
#             # plot
#             if surr_plot:
#                 fig = plt.figure(figsize=(10, 10))
#                 fig.patch.set_facecolor('xkcd:white')
#                 plt.title(labels_all[rc] + ' Surrogate Testing of CC LL', fontsize=20)
#                 plt.hist(LL_surr.reshape(-1), color=[0, 0, 0], alpha=0.3, label='surrogates, n: ' + str(len(LL_surr)))
#                 plt.xlim([0, np.max([np.nanpercentile(LL_surr, 99) * 1.5, 1])])
#                 plt.axvline(np.nanpercentile(LL_surr, 99), color=[1, 0, 0], label='99th')
#                 plt.axvline(np.nanpercentile(LL_surr, 95), color=[1, 0, 0], label='95th')
#                 plt.axvline(np.nanpercentile(LL_surr, 50), color=[0, 0, 0], label='50th')
#                 plt.legend(fontsize=15)
#                 plt.xticks(fontsize=15)
#                 plt.yticks(fontsize=15)
#                 plt.xlabel('LL of CC', fontsize=20)
#                 plt.ylabel('Number of Tests', fontsize=20)
#                 plt.savefig(fig_path + subj + '_' + labels_all[rc] + '_LL.svg')
#                 plt.savefig(fig_path + subj + '_' + labels_all[rc] + '.jpg')
#                 plt.close()
#
#         # np.save(file_t_resp, M_t_resp)
#         np.savez('mat.npz', name1=M_CC_LL_surr, name2=CC_WOI)
#         # np.save(file_CC_LL_surr, M_CC_LL_surr)
#         # np.save(file_CCp_surr, M_CCp_surr)
#         surr_thr = pd.DataFrame(np.concatenate([stims, M_CC_LL_surr], 1),
#                                 columns=['Chan', 'CC_LL50', 'CC_LL95', 'CC_LL99'])
#         surr_thr.to_csv(file_CC_surr,
#                         index=False,
#                         header=True)
#         update_sig_con = 1
#     # SUMMARY, M_GT_all = np.load(file_GT)
#     files_list = glob(path_patient_analysis + '\\' + folder + '/data/Stim_list_*')
#     stimlist = pd.read_csv(files_list[0])
#     lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
#     labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
#         stimlist,
#         lbls)
#     CC_summ = get_CC_summ(M_GT_all, M_t_resp, surr_thr, coord_all, t_0=1, w=0.25, Fs=500)
#     CC_summ.insert(0, 'Subj', subj)
#     CC_summ.to_csv(file_CC_summ, header=True, index=False)
#     # file_sig_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\sig_con.csv'
#
#     # arr = np.zeros((1, 9))
#     # for sc in range(len(M_t_resp)):
#     #     arr_sc = np.zeros((len(M_t_resp), 9))
#     #     arr_sc[:, 0] = sc
#     #     arr_sc[:, 1] = np.arange(len(M_t_resp))
#     #     arr_sc[:, 2:] = M_t_resp[sc, :, [0, 4, 1, 2, 3, 5, 6]].T
#     #     arr = np.concatenate([arr, arr_sc], 0)
#     # arr = arr[1:, :]
#     # sig_con = pd.DataFrame(arr,
#     #                        columns=['Stim', 'Chan', 'Sig_LL', 'Sig_CCp', 't_onset', 't_resp', 'CCp', 'CC_LL1',
#     #                                 'CC_LL2'])
#     # sig_con.loc[sig_con.Sig_LL == -1, 'Sig_CCp'] = -1
#     # if update_sig_con:
#     #     sig_con.insert(3, 'Sig_CC_LL', 0)
#     #     for rc in range(len(M_t_resp)):
#     #         thr = 1.1 * surr_thr.loc[surr_thr.Chan == rc, 'CC_LL99'].values[0]
#     #         sig_con.loc[(sig_con.Chan == rc) & ((sig_con.CC_LL1 >= thr) | (sig_con.CC_LL2 >= thr)), 'Sig_CC_LL'] = 1
#     #     sig_con.loc[sig_con.Sig_LL == -1, 'Sig_CC_LL'] = -1
#     #     sig_con.to_csv(file_sig_con,
#     #                    index=False,
#     #                    header=True)
#
#     print('Done')
#
#
# ##first you have to have con_trial_alll "EL010","EL011", "EL012",'EL013',
# for subj in [
#     "EL010"]:  # ,"EL011", "EL012",'EL013','EL014',"EL015","EL016","EL017" "EL010","EL011", "EL012",'EL013','EL014',"EL015","EL016","EL017" ## ,"EL011", "EL010", "EL012", 'EL014', "EL015", "EL016","EL017"
#     for f in ['BrainMapping']:  # 'BrainMapping', 'InputOutput',
#         # l = 0
#         # if subj =='EL011':
#         #    l = 1
#         start_subj_GT(subj, folder=f, cond_folder='CR', load_con=1, load_surr=0)
#
#         # thread = 0
#         # sig = 0
#         # for subj in ["EL016"]:  # 'EL015','EL014',
#         #
#         #     if thread:
#         #         _thread.start_new_thread(start_subj_GT, (subj,'BrainMapping', 'CR', 1))
#         #     else:
#         #         print('start -- ' + subj)
#         #         start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', rerun=1)
#         # if thread:
#         #     while 1:
#         time.sleep(1)
