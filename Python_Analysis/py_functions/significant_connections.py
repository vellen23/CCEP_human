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
import tqdm
import significance_funcs as sf
import time
import _thread
from pathlib import Path


def get_GT_trials(trials, real=1, t_0=1, Fs=500, w=0.25, n_cluster=2):
    # t_0 stimulation time in epoched data, changed in surrogate data
    # trials = EEG_resp[rc, stimNum_all, :], shape (n,2000)
    # 1. z-score trials and get LL for t_onset and t_max
    EEG_trial = ff.bp_filter(trials, 1, 40, Fs)
    # EEG_trial = stats.zscore(EEG_trial, axis=1)
    resp_mean = np.nanmean(EEG_trial, 0)  ## ff.lp_filter(np.nanmean(bf.zscore_CCEP(trials), 0), 40, Fs)
    LL_mean = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp_mean, 0), 0), Fs, w)
    # where LL is the highest
    t_resp = np.argmax(LL_mean[0, 0, int((t_0 + w / 2) * Fs):int((t_0 + 0.5 - w / 2) * Fs)]) / Fs
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
    # cluster trials based on specific window (where highest LL is)
    EEG_trial = stats.zscore(EEG_trial, axis=1)
    cc, y = Cf.ts_cluster(
        EEG_trial[:, int((t_0 + t_resp) * Fs):int((t_0 + t_resp + w) * Fs)], n=n_cluster,
        method='euclidean')
    # p_CC = np.corrcoef(cc[0], cc[1])[0, 1]  # pearson correlation of two clusters
    M_GT = np.zeros((3, trials.shape[-1]))
    M_GT[0, :] = ff.lp_filter(np.nanmean(trials, 0), 45, Fs)
    for c in range(n_cluster):
        M_GT[c + 1, :] = ff.lp_filter(np.nanmean(trials[y == c, :], 0), 45, Fs)
    p_CC = np.corrcoef(M_GT[1, int((t_0 + t_resp) * Fs):int((t_0 + t_resp + 2 * w) * Fs)],
                       M_GT[2, int((t_0 + t_resp) * Fs):int((t_0 + t_resp + 2 * w) * Fs)])[0, 1]
    LL_CC = LLf.get_LL_all(np.expand_dims(M_GT[1:, int((t_0 + t_resp) * Fs):int((t_0 + t_resp + w) * Fs)], 0), Fs, w)
    LL_CC = np.max(LL_CC[0, :, :], 1)
    return [r, t_onset, t_resp], LL_mean[0, 0], p_CC, M_GT, y, thr, LL_CC


def get_GT(sc, rc, LL_CCEP, EEG_resp, Fs=500, t_0=1, w_cluster=0.25, n_cluster=2):
    lists = LL_CCEP[(LL_CCEP['Artefact'] < 1) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
    stimNum_all = lists.Num.values.astype('int')
    M_GT = np.zeros((n_cluster + 1, 2000))
    r = -1
    t_onset = -1
    t_resp = -1
    p_CC = -1
    LL_CC = np.zeros((1, 2))
    if len(stimNum_all) > 0:
        t_nan = np.where(np.isnan(np.mean(EEG_resp[rc, stimNum_all, :], 1)) * 1)  # [0][0]
        if len(t_nan[0]) > 0:
            stimNum_all = np.delete(stimNum_all, t_nan[0])
        if len(stimNum_all) > 15:
            trials = EEG_resp[rc, stimNum_all, :]
            [r, t_onset, t_resp], _, p_CC, M_GT, y, _, LL_CC = get_GT_trials(trials, real=1, t_0=t_0, Fs=Fs,
                                                                             w=w_cluster,
                                                                             n_cluster=n_cluster)
    return M_GT, [r, t_onset, t_resp, p_CC], LL_CCEP, LL_CC


def get_CC_surr(rc, LL_CCEP, EEG_resp, n_trials, Fs=500, w_cluster=0.25, n_cluster=2):
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
    pear_surr = np.zeros((1,)) - 1
    LL_surr = np.zeros((1, 2))
    for i in range(200):
        StimNum_surr = np.random.choice(StimNum, size=n_trials).astype('int')
        t_nan = np.where(np.isnan(np.mean(EEG_resp[rc, StimNum_surr, :], 1)) * 1)  # [0][0]
        if len(t_nan[0]) > 0:
            StimNum_surr = np.delete(StimNum_surr, t_nan[0])
        # get overall mean and two clusters
        trials = EEG_resp[rc, StimNum_surr, :]
        trials[:, 0:1000] = EEG_resp[rc, StimNum_surr, 1000:]
        trials[:, 1000:] = 0.9 * np.flip(EEG_resp[rc, StimNum_surr, 1000:], 1)
        for t_0 in [0, 0.6, 1.1]:
            _, _, p_CC, _, y, _, LL_CC = get_GT_trials(trials, real=0, t_0=t_0, Fs=Fs, w=w_cluster,
                                                       n_cluster=n_cluster)
            if (sum(y==0)>np.max([5, 0.05*len(y)]))&(sum(y==1)>np.max([5, 0.05*len(y)])):
                pear_surr = np.concatenate([pear_surr, [p_CC]], 0)
                LL_surr = np.concatenate([LL_surr, np.expand_dims(LL_CC, 0)], 0)
    return pear_surr, [np.percentile(pear_surr, 95), np.percentile(pear_surr, 99)], LL_surr[1:, :]


def start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', load_con=1, load_surr=1, surr_plot=1):
    print(subj + ' -- START --')
    ## path_patient_analysis = 'T:\EL_experiment\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_patient_analysis = 'y:\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_gen = os.path.join('y:\\eLab\Patients\\' + subj)
    if not os.path.exists(path_gen):
        path_gen = 'T:\\EL_experiment\\Patients\\' + subj
    path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    path_infos = os.path.join(path_patient, 'infos')
    if not os.path.exists(path_infos):
        path_infos = path_gen + '\\infos'

    lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
    labels_all = lbls.label.values

    # files of interests
    file_t_resp = path_patient_analysis + '\\' + folder + '\\data\\M_tresp.npy'  # for each connection: LLsig (old), t_onset (old), t_resp, CC_p, CC_LL1, CC_LL2
    file_CC_surr = path_patient_analysis + '\\' + folder + '\\data\\M_CC_surr.csv'

    file_GT = path_patient_analysis + '\\' + folder + '\\data\\M_CC.npy'
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    con_trial = pd.read_csv(file_con)
    con_trial.loc[np.isnan(con_trial.P2P), 'Artefact'] = 1
    con_trial.loc[con_trial.LL == 0, 'Artefact'] = 1
    EEG_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy'
    # if not os.path.isfile(file_GT):
    ##### 1. get cluster centers and t_resp, t_onset for each possible connection
    EEG_resp = []
    if os.path.isfile(file_GT) * load_con:
        # print(file_GT + ' -- already exists')
        M_t_resp = np.load(file_t_resp)
        # M_GT_all = np.load(file_GT)
    else:
        if not os.path.isfile(EEG_CR_file):
            print('Run concat.py first to get concatenated EEG data')
            return
        else:
            print('loading EEG data ..... ')
            EEG_resp = np.load(EEG_CR_file)

        chan_all = np.unique(con_trial.Chan)
        n_chan = np.max(chan_all).astype('int') + 1
        M_GT_all = np.zeros((n_chan, n_chan, 3, 2000))  # mean and 3 Cluster Centers
        M_t_resp = np.zeros((n_chan, n_chan, 7))  # LL_sig, t_onset, t_resp, pearson of GT
        M_t_resp[:, :, 0] = -1  # sig_LL of mean
        M_t_resp[:, :, 4] = -1  # sig_CCp
        for sc in tqdm.tqdm(np.unique(con_trial.Stim)):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0) & (con_trial.Stim == sc), 'Chan']).astype(
                'int')
            for rc in resp_chans:
                M_GT_all[sc, rc, :, :], M_t_resp[sc, rc, :4], con_trial, M_t_resp[sc, rc, 5:] = get_GT(sc, rc,
                                                                                                       con_trial,
                                                                                                       EEG_resp)
        np.save(file_GT, M_GT_all)
        np.save(file_t_resp, M_t_resp)
        con_trial.to_csv(file_con,
                         index=False,
                         header=True)
        print(subj + ' -- DONE --')
    fig_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\methods\\CC_surr_hist\\'

    Path(fig_path).mkdir(
        parents=True, exist_ok=True)
    ###### 2. calculate for each recording channel surrogate CC pearson correlations
    if os.path.isfile(file_CC_surr) * load_surr:
        M_t_resp = np.load(file_t_resp)  ## LL_sig, t_onset, t_resp, pearson of GT
        update_sig_con = 0
    else:
        M_CCp_surr = np.zeros((len(labels_all), 2))
        M_CC_LL_surr = np.zeros((len(labels_all), 2))
        stims = np.zeros((len(labels_all), 1))
        stims[:, 0] = np.arange(len(labels_all))

        con_trial_n = con_trial[(con_trial.Artefact < 1) & (con_trial.LL > 0)]
        summ = con_trial_n.groupby(['Stim', 'Chan'], as_index=False)[['LL']].count()
        if len(EEG_resp) == 0:
            print('loading EEG data ..... ')
            EEG_resp = np.load(EEG_CR_file)
        resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0), 'Chan']).astype(
            'int')
        for rc in tqdm.tqdm(resp_chans):
            n_trials = np.median(summ.loc[summ.Chan == rc, 'LL']).astype('int')
            pear_surr, M_CCp_surr[rc, :], LL_surr = get_CC_surr(rc, con_trial, EEG_resp, n_trials)
            M_CC_LL_surr[rc, :] = [np.percentile(LL_surr, 95), np.percentile(LL_surr, 99)]
            # pearson's surr test
            M_t_resp[M_t_resp[:, rc, 3] >= M_CCp_surr[rc, 1], rc, 4] = 1
            M_t_resp[M_t_resp[:, rc, 3] < M_CCp_surr[rc, 1], rc, 4] = 0
            # plot
            if surr_plot:
                fig = plt.figure(figsize=(10, 10))
                fig.patch.set_facecolor('xkcd:white')
                plt.title(labels_all[rc] + ' Surrogate Testing of CC pearson Correlation', fontsize=20)
                plt.hist(pear_surr, color=[0, 0, 0], alpha=0.3, label='surrogates, n: ' + str(len(pear_surr)))
                plt.xlim([-1.1, 1.1])
                plt.axvline(np.percentile(pear_surr, 99), color=[1, 0, 0], label='99th')
                plt.axvline(np.percentile(pear_surr, 95), color=[1, 0, 0], label='95th')
                plt.legend(fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('Pearson Correlation CC1 - CC2', fontsize=20)
                plt.ylabel('Number of Tests', fontsize=20)
                plt.savefig(fig_path + subj + '_' + labels_all[rc] + '_pearson.svg')
                # plt.savefig(fig_path + subj + '_' + labels_all[rc] + '.jpg')
                plt.close()

                fig = plt.figure(figsize=(10, 10))
                fig.patch.set_facecolor('xkcd:white')
                plt.title(labels_all[rc] + ' Surrogate Testing of CC LL', fontsize=20)
                plt.hist(LL_surr.reshape(-1), color=[0, 0, 0], alpha=0.3, label='surrogates, n: ' + str(len(LL_surr)))
                plt.xlim([0, np.max([np.percentile(LL_surr, 99) * 1.5, 1])])
                plt.axvline(np.percentile(LL_surr, 99), color=[1, 0, 0], label='99th')
                plt.axvline(np.percentile(LL_surr, 95), color=[1, 0, 0], label='95th')
                plt.legend(fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('LL of CC', fontsize=20)
                plt.ylabel('Number of Tests', fontsize=20)
                plt.savefig(fig_path + subj + '_' + labels_all[rc] + '_LL.svg')
                # plt.savefig(fig_path + subj + '_' + labels_all[rc] + '.jpg')
                plt.close()

        np.save(file_t_resp, M_t_resp)
        # np.save(file_CC_LL_surr, M_CC_LL_surr)
        # np.save(file_CCp_surr, M_CCp_surr)
        surr_thr = pd.DataFrame(np.concatenate([stims, M_CCp_surr, M_CC_LL_surr], 1),
                                columns=['Chan', 'CCp95', 'CCp99', 'CC_LL95', 'CC_LL99'])
        surr_thr.to_csv(file_CC_surr,
                        index=False,
                        header=True)
        update_sig_con = 1

    file_sig_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\sig_con.csv'

    arr = np.zeros((1, 9))
    for sc in range(len(M_t_resp)):
        arr_sc = np.zeros((len(M_t_resp), 9))
        arr_sc[:, 0] = sc
        arr_sc[:, 1] = np.arange(len(M_t_resp))
        arr_sc[:, 2:] = M_t_resp[sc, :, [0, 4, 1, 2, 3, 5, 6]].T
        arr = np.concatenate([arr, arr_sc], 0)
    arr = arr[1:, :]
    sig_con = pd.DataFrame(arr,
                           columns=['Stim', 'Chan', 'Sig_LL', 'Sig_CCp', 't_onset', 't_resp', 'CCp', 'CC_LL1', 'CC_LL2'])
    sig_con.loc[sig_con.Sig_LL == -1, 'Sig_CCp'] = -1
    if update_sig_con:
        sig_con.insert(3, 'Sig_CC_LL', 0)
        for rc in range(len(M_t_resp)):
            thr = 1.1 * surr_thr.loc[surr_thr.Chan == rc, 'CC_LL99'].values[0]
            sig_con.loc[(sig_con.Chan == rc) & ((sig_con.CC_LL1 >= thr) | (sig_con.CC_LL2 >= thr)), 'Sig_CC_LL'] = 1
        sig_con.loc[sig_con.Sig_LL == -1, 'Sig_CC_LL'] = -1
        sig_con.to_csv(file_sig_con,
                       index=False,
                       header=True)


    print('Done')


##first you have to have con_trial_alll "EL010","EL011", "EL012",'EL013',
for subj in ["EL018"]:  #,"EL011", "EL012",'EL013','EL014',"EL015","EL016","EL017" "EL010","EL011", "EL012",'EL013','EL014',"EL015","EL016","EL017" ## ,"EL011", "EL010", "EL012", 'EL014', "EL015", "EL016","EL017"
    for f in ['BrainMapping']:  # 'BrainMapping', 'InputOutput',
        # l = 0
        # if subj =='EL011':
        #    l = 1
        start_subj_GT(subj, folder=f, cond_folder='CR', load_con=0, load_surr=0)

        # thread = 0
        # sig = 0
        # for subj in ["EL016"]:  # 'EL015','EL014',
        #
        #     if thread:
        #         _thread.start_new_thread(start_subj_GT, (subj,'BrainMapping', 'CR', 1))
        #     else:
        #         print('start -- ' + subj)
        #         start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', rerun=1)
        # if thread:
        #     while 1:
        #         time.sleep(1)
