import os
import numpy as np
import sys
import statsmodels
sys.path.append('T:\\EL_experiment\\Codes\\CCEP_human\\Python_Analysis\\py_functions')
import pandas as pd
from glob import glob
import basic_func as bf
import matplotlib.pyplot as plt
import tqdm
import significant_connections as SCF
import matplotlib.font_manager as fm
import h5py
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

def trial_significance(subj, folder='BrainMapping', cond_folder='CR', p=0.05):
    print(subj + ' ---- START ------ ')

    # path_patient_analysis = 'Y:\\eLab\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    ## get labels for each channel
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'

    ## load required data
    con_trial = pd.read_csv(
        file_con)  # table of each stimulation and for each response channel the corresponding LL value etc.
    # update sig threshold
    req = (con_trial.p_value_LL>=0)&(con_trial.Artefact<1)
    con_trial.loc[req, 'Sig'] = 0
    p_values =con_trial.loc[req, 'p_value_LL'].values
    p_sig, p_corr = statsmodels.stats.multitest.fdrcorrection(abs(p_values-1))
    con_trial.loc[req, 'Sig'] = np.array(p_sig*1)
    con_trial['Sig'] = pd.to_numeric(con_trial['Sig'], errors='coerce')
    con_trial.to_csv(file_con, header=True, index=False)

def start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method='kmeans', skipt_GT=1, skip_surr=1,skip_summ=1,
                  trial_sig_labeling=1):
    print(subj + ' ---- START ------ ')

    # path_patient_analysis = 'Y:\\eLab\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj

    path_gen = os.path.join(sub_path + '\\Patients\\' + subj)
    if not os.path.exists(path_gen):
        path_gen = 'T:\\EL_experiment\\Patients\\' + subj
    path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    path_infos = os.path.join(path_gen, 'Electrodes')
    stimlist = pd.read_csv(
        path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\stimlist_' + cond_folder + '.csv')
    ## get labels for each channel
    lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
    if "type" in lbls:
        lbls = lbls[lbls.type == 'SEEG']
        lbls = lbls.reset_index(drop=True)
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist,
        lbls)
    badchans = pd.read_csv(path_patient_analysis + '/BrainMapping/data/badchan.csv')
    bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])

    bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]

    ## files of interests
    file_t_resp = path_patient_analysis + '\\' + folder + '\\data\\M_tresp.npy'  # for each connection: LLsig (old), t_onset (old), t_resp, CC_p, CC_LL1, CC_LL2
    file_CC_surr = path_patient_analysis + '\\' + folder + '\\data\\M_CC_surr_' + cluster_method + '.csv'
    file_CC_LL_surr = path_patient_analysis + '\\' + folder + '\\data\\LL_CC_surr_' + cluster_method + '.h5'
    file_GT = path_patient_analysis + '\\' + folder + '\\data\\M_CC_' + cluster_method + '.h5'
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    file_CC_summ = path_patient_analysis + '\\' + folder + '\\data\\CC_summ_' + cluster_method + '.csv'

    ## load required data
    con_trial = pd.read_csv(
        file_con)  # table of each stimulation and for each response channel the corresponding LL value etc.
    # trials where there is no LL value get an aretfact label (removed from analysis)
    con_trial.loc[con_trial.LL == 0, 'Artefact'] = 1
    # file where epoched data is stored (chan, trial, 2000)
    EEG_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.h5'
    if os.path.isfile(EEG_CR_file):
        h5 = 1
    else:
        EEG_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy'
        h5 = 0

    ##### 1. get cluster centers and t_resp, t_onset for each possible connection
    EEG_resp = []
    M_GT_all = []
    if os.path.isfile(file_GT) * skipt_GT:
        # print(file_GT + ' -- already exists')
        M_t_resp = np.load(file_t_resp)
        # M_GT_all = np.load(file_GT)
    else:
        if not os.path.isfile(EEG_CR_file):
            print('Run concat.py first to get concatenated EEG data')
            return
        else:
            print('loading EEG data ..... ')
            if h5:
                print('loading h5')
                EEG_resp = h5py.File(EEG_CR_file)
                EEG_resp = EEG_resp['EEG_resp']
            else:
                print('loading npy')
                EEG_resp = np.load(EEG_CR_file)
        print('Calculating CC for each connection ..... ')
        chan_all = np.unique(con_trial.Chan)
        n_chan = np.max(chan_all).astype('int') + 1  # number of channels. should be the same are len(labels_all)
        M_GT_all = np.zeros((n_chan, n_chan, 3, 2000))  # mean and 2 Cluster Centers
        M_t_resp = np.zeros(
            (n_chan, n_chan, 6))  # LL_sig (general), t_onset (general), t_WOI,2x LL of WOI of CC, LL of mean
        M_t_resp[:, :, 0] = -1  # sig_LL of mean
        for sc in tqdm.tqdm(np.unique(con_trial.Stim)):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Ictal == 0) &(con_trial.Artefact < 1) & (con_trial.Stim == sc), 'Chan']).astype(
                'int')
            for rc in resp_chans:
                # GT output: M_GT, [r, t_onset, t_WOI], LL_CC
                M_GT_all[sc, rc, :, :], M_t_resp[sc, rc, :3], M_t_resp[sc, rc, 3:5], M_t_resp[sc, rc, 5] = SCF.get_GT(
                    sc, rc,
                    con_trial,
                    EEG_resp)  ## CC
        # np.save(file_GT, M_GT_all)
        with h5py.File(file_GT, 'w') as hf:
            hf.create_dataset("M_GT_all", data=M_GT_all)

        np.save(file_t_resp, M_t_resp)

        print(subj + ' -- CC calculation DONE --', end='\r')
    fig_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\methods\\CC_surr_hist_' + cluster_method + '\\'

    Path(fig_path).mkdir(
        parents=True, exist_ok=True)
    ###### 2. calculate for each recording channel surrogate CC
    n_surr = 200
    if os.path.isfile(file_CC_surr) * skip_surr:
        M_t_resp = np.load(file_t_resp)  ## LL_sig, t_onset, t_resp, pearson of GT
        surr_thr = pd.read_csv(file_CC_surr)
        update_sig_con = 0
    else:
        M_CC_LL_surr = np.zeros((len(labels_all), 6))
        CC_LL_surr = np.zeros((len(labels_all), n_surr, 2, 2000))
        CC_WOI = np.zeros((len(labels_all), n_surr))
        stims = np.zeros((len(labels_all), 1))
        stims[:, 0] = np.arange(len(labels_all))

        con_trial_n = con_trial[(con_trial.Artefact < 1) & (con_trial.LL > 0)]
        summ = con_trial_n.groupby(['Stim', 'Chan'], as_index=False)[['LL']].count()
        if len(EEG_resp) == 0:
            print('loading EEG data ..... ', end='\r')
            if h5:
                print('loading h5', end='\r')
                EEG_resp = h5py.File(EEG_CR_file)
                EEG_resp = EEG_resp['EEG_resp']
            else:
                print('loading npy', end='\r')
                EEG_resp = np.load(EEG_CR_file)
        resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0), 'Chan']).astype(
            'int')
        for rc in tqdm.tqdm(resp_chans):
            n_trials = np.median(summ.loc[summ.Chan == rc, 'LL']).astype('int')
            LL_surr, CC_LL_surr[rc], CC_WOI[rc], LL_mean_surr = SCF.get_CC_surr(rc, con_trial, EEG_resp,
                                                                                n_trials)  # return LL_surr[1:, :], LL_surr_data[1:, :, :]
            # LL_CC_surr, LL_surr_data, WOI_surr, LL_mean_surr
            M_CC_LL_surr[rc, :] = [np.nanpercentile(LL_surr, 50), np.nanpercentile(LL_surr, 95),
                                   np.nanpercentile(LL_surr, 99), np.nanpercentile(LL_mean_surr, 50),
                                   np.nanpercentile(LL_mean_surr, 95),
                                   np.nanpercentile(LL_mean_surr, 99)]

        with h5py.File(file_CC_LL_surr, 'w') as hf:
            hf.create_dataset("CC_LL_surr", data=CC_LL_surr)
            hf.create_dataset("CC_WOI", data=CC_WOI)

        # np.save(file_CC_LL_surr, M_CC_LL_surr)
        surr_thr = pd.DataFrame(np.concatenate([stims, M_CC_LL_surr], 1),
                                columns=['Chan', 'CC_LL50', 'CC_LL95', 'CC_LL99', 'mean_LL50', 'mean_LL95',
                                         'mean_LL99'])
        surr_thr.to_csv(file_CC_surr,
                        index=False,
                        header=True)

        update_sig_con = 1
        print(subj + ' -- CC surrogate calculation DONE --', end='\r')
    # SUMMARY
    if os.path.isfile(file_CC_summ) * skip_surr * skipt_GT*skip_summ:
        CC_summ = pd.read_csv(file_CC_summ)
    else:
        files_list = glob(path_patient_analysis + '\\' + folder + '/data/Stim_list_*')
        stimlist = pd.read_csv(files_list[0])
        lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
        labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
            stimlist,
            lbls)
        if len(M_GT_all) == 0:
            # M_GT_all = np.load(file_GT)
            M_GT_all = h5py.File(file_GT)
            M_GT_all = M_GT_all['M_GT_all']

            CC_LL_surr = h5py.File(file_CC_LL_surr)
            CC_LL_surr = CC_LL_surr['CC_LL_surr']

        CC_summ = SCF.get_CC_summ(M_GT_all, M_t_resp, CC_LL_surr, surr_thr, coord_all, t_0=1, w=0.25, Fs=500)
        CC_summ.insert(0, 'Subj', subj)
        CC_summ.to_csv(file_CC_summ, header=True, index=False)
        print(subj + ' -- CC summary saved --', end='\r')

    if trial_sig_labeling:
        if len(M_GT_all) == 0:
            print('loading GT data ..... ', end='\r')
            M_GT_all = h5py.File(file_GT)
            M_GT_all = M_GT_all['M_GT_all']

        if len(EEG_resp) == 0:
            print('loading EEG data ..... ', end='\r')
            if h5:
                print('loading h5', end='\r')
                EEG_resp = h5py.File(EEG_CR_file)
                EEG_resp = EEG_resp['EEG_resp']
            else:
                print('loading npy', end='\r')
                EEG_resp = np.load(EEG_CR_file)
        new_col = ['Sig', 'LL_WOI', 'LL_pre']
        for col in new_col:
            if col in con_trial:
                con_trial = con_trial.drop(columns=col)
            con_trial.insert(5, col, -1)
        del_col = ['t_N2', 't_N1', 'sN2', 'sN1', 'N2', 'N1']
        for col in del_col:
            if col in con_trial:
                con_trial = con_trial.drop(columns=col)
        print('Get sig trial label....')
        CC_summ["sig"] = 0
        p = abs(CC_summ.p_val.values-1)
        p_sig, _ = statsmodels.stats.multitest.fdrcorrection(p)
        CC_summ['sig'] = np.array(p_sig * 1)
        CC_summ['sig'] = pd.to_numeric(CC_summ['sig'], errors='coerce')
        for sc in tqdm.tqdm(np.unique(con_trial.Stim), desc='Stimulation Channel'):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Artefact < 1) & (con_trial.Stim == sc), 'Chan']).astype(
                'int')
            for rc in resp_chans:
                # decide for sig threhsold (only CC or also mean)
                dat = CC_summ.loc[
                    (CC_summ.Stim == sc) & (CC_summ.Chan == rc) & (CC_summ.sig == 1) & (CC_summ.sig_w == 1)]  # & (CC_summ.art == 0)
                # if there is a significant CC in this connection
                if len(dat) > 0:
                    ix_cc = dat.CC.values.astype('int')
                    ix_cc = np.concatenate([np.array([0]), ix_cc])
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
        print(subj + ' -- single trial DONE --', end='\r')
        rem_art = 1
        if rem_art:
            con_trial.loc[(con_trial.Artefact == 3), 'Artefact'] = 0
            con_trial = mark_artefacts(con_trial, 'LL_pre')
            con_trial.to_csv(file_con,
                             index=False,
                             header=True)
            print(subj + ' -- bad trial removed --', end='\r')

    get_SNR = 0

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
        print(subj + ' -- SNR DONE --', end='\r')

    print(subj + ' -- all DONE --')


def mark_artefacts(con_trial, metric):
    from scipy.stats import zscore
    # group['z_score'] = zscore(group['P2P_BL'])
    # group['Artefact'] = group['z_score'].apply(lambda x: 1 if x > 6 else group['Artefact'])
    for c in np.unique(con_trial.Chan):
        val_dist = con_trial.loc[(con_trial.Chan==c)&((con_trial.Artefact==0)), metric].values
        val_dist_z = (val_dist-np.nanmean(val_dist))/np.nanstd(val_dist)
        con_trial.loc[(con_trial.Chan==c)&((con_trial.Artefact==0)), 'zscore'] = val_dist_z
        con_trial.loc[(con_trial.Chan == c) & (con_trial.Artefact == 0)& (con_trial.LL_pre >12)& (con_trial.zscore >8), 'Artefact'] = 3
        con_trial.loc[(con_trial.Chan == c) & (con_trial.Artefact == 0) & (con_trial.P2P_BL > 3000) & (
                    con_trial.zscore > 8), 'Artefact'] = 3
    con_trial.drop('zscore', axis = 1, inplace = True)
    return con_trial

def sig_con_keller(subj, folder='BrainMapping', cond_folder='CR', t0=1, Fs=500):
    import CCEP_func
    print(subj + ' ---- START ------ ')

    # path_patient_analysis = 'Y:\\eLab\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    # 1. get summary table
    file_CC_summ = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\summ_general.csv'  # summary_genera
    con_summary_all = pd.read_csv(file_CC_summ)
    con_summary_all = con_summary_all.drop_duplicates()
    con_summary_all['Zscore'] = np.nan # con_summary_all.insert(5, 'Zscore', 0)
    file_con = path_patient_analysis + '\\' + folder + '/' + cond_folder + '/data/con_trial_all.csv'
    con_trial = pd.read_csv(file_con)
    # 2. get EEG data
    h5_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.h5'
    if os.path.isfile(h5_file):
        print('loading h5')
        EEG_resp = h5py.File(h5_file)
        EEG_resp = EEG_resp['EEG_resp']

        for sc in np.unique(con_summary_all['Stim']):
            for rc in np.unique(con_summary_all.loc[(con_summary_all.Stim == sc), 'Chan']):
                num_all = np.unique(con_trial.loc[(con_trial.Chan == rc) & (con_trial.Stim == sc)& (con_trial.Artefact <1), 'Num'].values)
                resp_zscore_mean = CCEP_func.zscore_CCEP(np.mean(EEG_resp[rc, num_all], 0), t_0=1, w0=0.5, Fs=Fs)

                # Calculate max zscore for each response channel in the specified time window
                zscore_pk = np.max(abs(resp_zscore_mean[int((t0 + 0.05) * Fs):int((t0 + 0.5) * Fs)]))
                con_summary_all.loc[(con_summary_all['Stim'] == sc) & (con_summary_all['Chan'] == rc), 'Zscore'] = \
                    zscore_pk
    con_summary_all = con_summary_all.drop_duplicates()
    con_summary_all.to_csv(file_CC_summ, header=True, index=False)
