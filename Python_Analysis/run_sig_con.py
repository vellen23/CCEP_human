import os
import numpy as np
import sys

sys.path.append('T:\\EL_experiment\\Codes\\CCEP_human\\Python_Analysis\\py_functions')
import pandas as pd
from glob import glob
import basic_func as bf
import matplotlib.pyplot as plt
import tqdm
import significant_connections as SCF
import significance_funcs as sigf

from pathlib import Path

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', load_con=1, load_surr=1, surr_plot=1):
    print(subj + ' -- START --')
    ## define path of your analysis
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_gen = os.path.join(sub_path, 'Patients', subj)
    if not os.path.exists(path_gen):
        print("Can't find path")
        # path_gen = 'T:\\EL_experiment\\Patients\\' + subj
        return
    path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    path_infos = os.path.join(path_patient, 'infos')
    if not os.path.exists(path_infos):
        path_infos = path_gen + '\\infos'
    ## get labels for each channel
    lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
    labels_all = lbls.label.values

    ## files of interests
    file_t_resp = path_patient_analysis + '\\' + folder + '\\data\\M_tresp.npy'  # for each connection: LLsig (old), t_onset (old), t_resp, CC_p, CC_LL1, CC_LL2
    file_CC_surr = path_patient_analysis + '\\' + folder + '\\data\\M_CC_surr.csv'
    file_CC_LL_surr = path_patient_analysis + '\\' + folder + '\\data\\LL_CC_surr.npz'
    file_GT = path_patient_analysis + '\\' + folder + '\\data\\M_CC.npy'
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    file_CC_summ = path_patient_analysis + '\\' + folder + '\\data\\CC_summ.csv'

    ## load required data
    con_trial = pd.read_csv(
        file_con)  # table of each stimulation and for each response channel the corresponding LL value etc.
    # trials where there is no LL value get an aretfact label (removed from analysis)
    con_trial.loc[con_trial.LL == 0, 'Artefact'] = 1
    # file where epoched data is stored (chan, trial, 2000)
    EEG_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy'

    ##### 1. get cluster centers and t_resp, t_onset for each possible connection
    EEG_resp = []
    M_GT_all = []
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
        print('Calculating CC for each connection ..... ')
        chan_all = np.unique(con_trial.Chan)
        n_chan = np.max(chan_all).astype('int') + 1  # number of channels. should be the same are len(labels_all)
        M_GT_all = np.zeros((n_chan, n_chan, 3, 2000))  # mean and 2 Cluster Centers
        M_t_resp = np.zeros(
            (n_chan, n_chan, 5))  # LL_sig (general), t_onset (general), t_WOI,2x LL of WOI of CC, mean of BL
        M_t_resp[:, :, 0] = -1  # sig_LL of mean
        for sc in tqdm.tqdm(np.unique(con_trial.Stim)):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0) & (con_trial.Stim == sc), 'Chan']).astype(
                'int')
            for rc in resp_chans:
                # GT output: M_GT, [r, t_onset, t_WOI], LL_CC
                M_GT_all[sc, rc, :, :], M_t_resp[sc, rc, :3], M_t_resp[sc, rc, 3:] = SCF.get_GT(sc, rc,
                                                                                                con_trial,
                                                                                                EEG_resp)
        np.save(file_GT, M_GT_all)
        np.save(file_t_resp, M_t_resp)

        print(subj + ' -- DONE --')
    fig_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\methods\\CC_surr_hist\\'

    Path(fig_path).mkdir(
        parents=True, exist_ok=True)
    ###### 2. calculate for each recording channel surrogate CC
    n_surr = 200
    if os.path.isfile(file_CC_surr) * load_surr:
        M_t_resp = np.load(file_t_resp)  ## LL_sig, t_onset, t_resp, pearson of GT
        surr_thr = pd.read_csv(file_CC_surr)
        update_sig_con = 0
    else:
        # M_CCp_surr = np.zeros((len(labels_all), 2))
        M_CC_LL_surr = np.zeros((len(labels_all), 3))
        CC_LL_surr = np.zeros((len(labels_all), n_surr, 2, 2000))
        CC_WOI = np.zeros((len(labels_all), n_surr))
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
            LL_surr, CC_LL_surr[rc], CC_WOI[rc] = SCF.get_CC_surr(rc, con_trial, EEG_resp,
                                                                  n_trials)  # return LL_surr[1:, :], LL_surr_data[1:, :, :]
            M_CC_LL_surr[rc, :] = [np.nanpercentile(LL_surr, 50), np.nanpercentile(LL_surr, 95),
                                   np.nanpercentile(LL_surr, 99)]

            # plot
            surr_plot = 0
            if surr_plot:
                fig = plt.figure(figsize=(10, 10))
                fig.patch.set_facecolor('xkcd:white')
                plt.title(labels_all[rc] + ' Surrogate Testing of CC LL', fontsize=20)
                plt.hist(LL_surr.reshape(-1), color=[0, 0, 0], alpha=0.3, label='surrogates, n: ' + str(len(LL_surr)))
                plt.xlim([0, np.max([np.nanpercentile(LL_surr, 99) * 1.5, 1])])
                plt.axvline(np.nanpercentile(LL_surr, 99), color=[1, 0, 0], label='99th')
                plt.axvline(np.nanpercentile(LL_surr, 95), color=[1, 0, 0], label='95th')
                plt.axvline(np.nanpercentile(LL_surr, 50), color=[0, 0, 0], label='50th')
                plt.legend(fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('LL of CC', fontsize=20)
                plt.ylabel('Number of Tests', fontsize=20)
                plt.savefig(fig_path + subj + '_' + labels_all[rc] + '_LL.svg')
                plt.savefig(fig_path + subj + '_' + labels_all[rc] + '.jpg')
                plt.close()

        np.savez(file_CC_LL_surr, name1=CC_LL_surr, name2=CC_WOI)
        # np.save(file_CC_LL_surr, M_CC_LL_surr)
        surr_thr = pd.DataFrame(np.concatenate([stims, M_CC_LL_surr], 1),
                                columns=['Chan', 'CC_LL50', 'CC_LL95', 'CC_LL99'])
        surr_thr.to_csv(file_CC_surr,
                        index=False,
                        header=True)
        update_sig_con = 1

    # SUMMARY
    if os.path.isfile(file_CC_summ):
        CC_summ = pd.read_csv(file_CC_summ)
    else:
        files_list = glob(path_patient_analysis + '\\' + folder + '/data/Stim_list_*')
        stimlist = pd.read_csv(files_list[0])
        lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
        labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
            stimlist,
            lbls)
        if len(M_GT_all) == 0:
            print('loading EEG data ..... ')
            M_GT_all = np.load(file_GT)

        CC_summ = SCF.get_CC_summ(M_GT_all, M_t_resp, surr_thr, coord_all, t_0=1, w=0.25, Fs=500)
        CC_summ.insert(0, 'Subj', subj)
        CC_summ.to_csv(file_CC_summ, header=True, index=False)

    trial_sig_labeling = 1
    if trial_sig_labeling:
        if len(M_GT_all) == 0:
            print('loading GT data ..... ')
            M_GT_all = np.load(file_GT)

        if len(EEG_resp) == 0:
            print('loading EEG data ..... ')
            EEG_resp = np.load(EEG_CR_file)
        new_col = ['Sig', 'LL_WOI', 'LL_pre']
        for col in new_col:
            if col in con_trial:
                con_trial = con_trial.drop(columns=col)
            con_trial.insert(5, col, -1)

        for sc in tqdm.tqdm(np.unique(con_trial.Stim), desc='Stimulation Channel'):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0) & (con_trial.Stim == sc), 'Chan']).astype(
                'int')
            for rc in resp_chans:
                dat = CC_summ.loc[
                    (CC_summ.Stim == sc) & (CC_summ.Chan == rc) & (CC_summ.sig_w == 1) & (
                                CC_summ.art == 0)]  # & (CC_summ.art == 0)
                # if there is a significant CC in this connection
                if len(dat) > 0:
                    ix_cc = dat.CC.values.astype('int')
                    M_GT = M_GT_all[sc, rc, ix_cc, :]
                    t_WOI = dat.t_WOI.values[0]
                    con_trial = SCF.get_sig_trial(sc, rc, con_trial, M_GT, t_WOI, EEG_resp, p=90, exp=2,
                                                  w_cluster=0.25)
                    # get_sig_trial(sc, rc, con_trial, M_GT, t_resp, EEG_CR, p=95, exp=2, w_cluster=0.25, t_0=1, Fs=500)
                else:
                    con_trial = SCF.get_sig_trial(sc, rc, con_trial, M_GT, t_WOI, EEG_resp, test=0, p=90, exp=2,
                                                  w_cluster=0.25)
                    con_trial.loc[(con_trial.Chan == rc) & (con_trial.Stim == sc), 'Sig'] = 0
        con_trial.to_csv(file_con,
                         index=False,
                         header=True)
    get_SNR = 1

    if get_SNR:
        con_trial.loc[con_trial.Artefact != 0, 'LL_pre'] = np.nan
        con_trial_SNR = con_trial.groupby(['Stim', 'Chan'], as_index=False)[['LL_pre']].mean()
        CC_summ.insert(4, 'SNR', np.nan)
        for sc in tqdm.tqdm(np.unique(con_trial.Stim)):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0) & (con_trial.Stim == sc), 'Chan']).astype(
                'int')
            for rc in resp_chans:
                m = con_trial_SNR.loc[(con_trial_SNR.Stim == sc) & (con_trial_SNR.Chan == rc), 'LL_pre'].values[0]
                CC_summ.loc[(CC_summ.Stim == sc) & (CC_summ.Chan == rc), 'SNR'] = CC_summ.loc[(CC_summ.Stim == sc) & (
                        CC_summ.Chan == rc), 'LL_WOI'].values / m
        CC_summ.to_csv(file_CC_summ, header=True, index=False)
    print('Done')


for subj in [
    "EL020"]:  # "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016", "EL017", "EL019","EL020"
    for f in ['BrainMapping']:  # 'BrainMapping', 'InputOutput',
        start_subj_GT(subj, folder=f, cond_folder='CR', load_con=1, load_surr=1)
