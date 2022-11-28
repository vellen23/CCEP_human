import os
import numpy as np
import pandas as pd
import sys
import tqdm
sys.path.append('T:\EL_experiment\Codes\CCEP_human\Python_Analysis/py_functions')
pd.options.mode.chained_assignment = None

dist_groups = np.array([[0, 30], [30, 60], [60, 120]])
dist_labels = ['local (<30 mm)', 'short (<60mm)', 'long']
Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3

folder = 'InputOutput'
# dur[0,:]       = np.int32(np.sum(abs(dur)))
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])
cwd = os.getcwd()

subjs = ['EL014', 'EL016', 'EL017'] # 'EL010', 'EL011', 'EL012', 'EL013', 'EL015',
for subj in subjs:
    print('Start --- ', subj)
    cond_folder = 'CR'  # Condition = 'Hour', 'Condition', 'Ph'

    path_patient_analysis = 'y:\\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_gen = os.path.join('y:\\eLab\Patients\\' + subj)
    if not os.path.exists(path_gen):
        path_gen = 'T:\\EL_experiment\\Patients\\' + subj
    path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    path_infos = os.path.join(path_patient, 'infos')
    if not os.path.exists(path_infos):
        path_infos = path_gen + '\\infos'
    file_con = path_patient_analysis + '\\' + folder + '/' + cond_folder + '/data/con_trial_all.csv'
    file_auc = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\AUC_connection.csv'
    file_bm_auc = path_patient_analysis + '\\BrainMapping\\' + cond_folder + '\\data\\BM_AUC_connection.csv'

    con_trial = pd.read_csv(file_con)
    con_trial = con_trial[con_trial.LL > 0]
    if not 'SleepState' in con_trial:
        con_trial.insert(5, 'SleepState', 'Wake')
        # con_trial.loc[(con_trial.Sleep == 0) , 'SleepState'] = 'W'
        con_trial.loc[(con_trial.Sleep > 0) & (con_trial.Sleep < 4), 'SleepState'] = 'NREM'
        con_trial.loc[(con_trial.Sleep == 4), 'SleepState'] = 'REM'
        con_trial.to_csv(file_con,
                         index=False,
                         header=True)
    AUC_all = np.zeros((1, 5))
    if not os.path.exists(file_auc):
        for sc in np.unique(con_trial.Stim).astype('int'):
            for rc in tqdm.tqdm(np.unique(con_trial.Chan).astype('int')):
                dat = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc)]
                if len(dat) > 0:
                    Int_all = np.unique(dat.Int)
                    LL = dat.groupby(['Int'])['LL'].mean()  # /LL_max

                    AUC_real = np.trapz(LL.values - LL.values[0], Int_all)

                    n_surr = 100
                    AUC_surr = np.zeros((n_surr,))
                    for i in range(n_surr):
                        dat_surr = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc)].copy()
                        random_ix = np.arange(len(dat_surr))
                        np.random.shuffle(random_ix)
                        dat_surr['Int'] = dat_surr['Int'].values[random_ix]# np.random.choice(dat_surr.loc[dat_surr.Stim==sc,'Int'].values, len(dat_surr.Int.values))
                        LL_surr = dat_surr.groupby(['Int'])['LL'].mean()  # /LL_max
                        AUC_surr[i] = np.trapz(LL_surr.values - LL_surr.values[0], Int_all)
                    thr = np.percentile(AUC_surr, 95)
                    if AUC_real > thr:
                        LL_min = np.min(dat.groupby(['Int', 'SleepState'])['LL'].mean())
                        dat = con_trial[(con_trial.Int == 12) & (con_trial.Stim == sc) & (con_trial.Chan == rc)]
                        LL_max = np.max(dat.groupby(['Int', 'SleepState'])['LL'].mean())

                        AUC1 = np.trapz(np.repeat(LL_max, len(Int_all)) - LL_min, Int_all)

                        for cond in np.unique(con_trial.SleepState):
                            dat = con_trial[
                                (con_trial.Stim == sc) & (con_trial.Chan == rc) & (
                                            con_trial.SleepState == cond) & ~np.isnan(
                                    con_trial.LL)]
                            if len(dat) > 0:
                                LL = dat.groupby(['Int'])['LL'].mean()  # /LL_max
                                AUC = np.trapz(LL - LL_min, np.unique(
                                    dat.Int.values)) / AUC1  # sklearn.metrics.auc(np.unique(dat.Int), LL-LL_min/LL_max)
                                v = np.max(dat.Int)
                                auc_sing = [[sc, rc, AUC, cond, LL[3] / LL[v]]]
                                AUC_all = np.concatenate([AUC_all, auc_sing])
                        dat = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc) & ~np.isnan(con_trial.LL)]
                        if len(dat) > 0:
                            LL = dat.groupby(['Int'])['LL'].mean()  # /LL_max
                            AUC = np.trapz(LL - LL_min, np.unique(
                                dat.Int.values)) / AUC1  # sklearn.metrics.auc(np.unique(dat.Int), LL-LL_min/LL_max)
                            v = np.max(dat.Int)
                            auc_sing = [[sc, rc, AUC, 'All', LL[3] / LL[v]]]
                            AUC_all = np.concatenate([AUC_all, auc_sing])

        AUC_all = AUC_all[1:, :]
        AUC_all = pd.DataFrame(AUC_all, columns=['Stim', 'Chan', 'AUC', 'SleepState', 'LL_int'])
        for cc in ['Stim', 'Chan', 'AUC', 'LL_int']:
            AUC_all[cc] = pd.to_numeric(AUC_all[cc])
        # AUC_all.insert(0,'Resp_Area','hipp')
        # AUC_all.insert(0,'Stim_Area','hipp')
        # for sc in np.unique(AUC_all[['Stim', 'Chan']]).astype('int'):
        #     AUC_all.loc[AUC_all.Stim==sc,'Stim_Area'] =lbls.Area.values[sc]
        #     AUC_all.loc[AUC_all.Chan==sc,'Resp_Area'] =lbls.Area.values[sc]
        # AUC_all = AUC_all[AUC_all.AUC>0]
        AUC_all.to_csv(file_auc, header=True, index=False)
    else:
        AUC_all = pd.read_csv(file_auc)
    ##merging
    # BM data
    if not os.path.exists(file_bm_auc):
        file_con = path_patient_analysis + '\\BrainMapping\\' + cond_folder + '/data/con_trial_all.csv'
        con_trial_BM = pd.read_csv(file_con)

        con_trial_sig = con_trial_BM[(con_trial_BM.Sig > -1) & (con_trial_BM.d > -10)]
        con_trial_sig.insert(4, 'LL_sig', np.nan)
        con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL_sig'] = con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL']
        summ = con_trial_sig[(con_trial_sig.Sig > -1)]  # only possible connections
        summ_AUC = summ.groupby(['Stim', 'Chan', 'SleepState'], as_index=False)[['Sig', 'LL_sig', 'd']].mean()
        summ_all = summ_AUC[np.isin(summ_AUC.Stim, np.unique(AUC_all.Stim))]

        #
        summ_AUC = summ.groupby(['Stim', 'Chan'], as_index=False)[['Sig', 'LL_sig', 'd']].mean()
        summ_AUC.insert(4, 'SleepState', 'All')
        summ_sleep = summ_AUC[np.isin(summ_AUC.Stim, np.unique(AUC_all.Stim))]
        BM_all = pd.concat([summ_sleep, summ_all])
        data_AUC_Prob = pd.merge(BM_all, AUC_all, on=['Stim', 'Chan', 'SleepState'])

        data_AUC_Prob.to_csv(file_bm_auc, header=True, index=False)
