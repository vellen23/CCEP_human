import sys

sys.path.append('./py_functions')
sys.path.append('./PCI/')
sys.path.append('./PCI/PCIst')
import os
import time
import _thread
import numpy as np
import pandas as pd
from tkinter import filedialog
from tkinter import *
import matplotlib.pyplot as plt
from pathlib import Path

root = Tk()
root.withdraw()
import scipy
from scipy import signal
import pylab
import scipy.cluster.hierarchy as sch
import platform
from glob import glob
import basic_func as bf
import IO_func as IOf
import LL_funcs as LLf
import NMF_funcs as NMFf
from sklearn.decomposition import NMF

##all
cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]

# plot
Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3

# dur[0,:]       = np.int32(np.sum(abs(dur)))
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
color_elab = np.zeros((4, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])
color_elab[3, :] = np.array([1, 0.574, 0])

x_ax_LL = np.arange(0, 0.5, (1 / Fs))
x_ax_LL_bad = np.arange(-1, 0, (1 / Fs))

## general
cwd = os.getcwd()
#
folder = 'InputOutput'
cond_folder = 'CR'
if platform.system() == 'Windows':
    # sep = ','
    path = 'Y:\\eLab\\Patients\\'  # + subj
    # path_patient_analysis = 'T:\EL_experiment\Projects\EL_experiment\Analysis\Patients\\' + subj
    # path_patient = 'T:\EL_experiment\Patients\\' + subj + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    CR_color = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\Analysis\BrainMapping\CR_color.xlsx",
                             header=0)
    regions = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\elab_labels.xlsx", sheet_name='regions',
                            header=0)

    # path_patient    = 'E:\PhD\EL_experiment\Patients\\'+subj # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
else:  # 'Darwin' for MAC
    path = '/Volumes/EvM_T7/PhD/EL_experiment/Patients/'  # + subj
    CR_color = pd.read_excel("/Volumes/EvM_T7/PhD/EL_experiment/Patients/all/Analysis/BrainMapping/CR_color.xlsx",
                             header=0)
    regions = pd.read_excel("/Volumes/EvM_T7/PhD/EL_experiment/Patients/all/elab_labels.xlsx", sheet_name='regions',
                            header=0)

sep = ';'
color_regions = regions.color.values
C = regions.label.values
cond_folder = 'CR'


def compute_subj(subj, metric='LL'):
    print(f'Performing calculations on {subj}')

    ######## General Infos
    path_patient_analysis = 'y:\\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_gen = os.path.join('y:\\eLab\Patients\\' + subj)
    if not os.path.exists(path_gen):
        path_gen = 'T:\\EL_experiment\\Patients\\' + subj
    path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    path_infos = os.path.join(path_patient, 'infos')
    if not os.path.exists(path_infos):
        path_infos = path_gen + '\\infos'

    if os.path.isfile(path_patient_analysis + '/InputOutput/' + cond_folder + '/data/EEG_' + cond_folder + '.npy'):
        # EEG_resp = np.load(path_patient + '/Analysis/InputOutput/'+cond_folder+'/data/EEG_'+cond_folder+'.npy')
        stimlist = pd.read_csv(
            path_patient_analysis + '/InputOutput/' + cond_folder + '/data/stimlist_' + cond_folder + '.csv')
    # elif os.path.isfile(path_patient + '/Analysis/InputOutput/' + cond_folder + '/data/EEG_' + cond_folder + '.npy')
    else:
        files_list = glob(path_patient_analysis + '/InputOutput/data/Stim_list_*_CR*')
        i = 0
        stimlist = pd.read_csv(files_list[i])
        # print('NOT FOUND --- ' + path_patient + '/Analysis/InputOutput/' + cond_folder + '/data/EEG_' + cond_folder + '.npy')
        # break
    # print('data loaded with ' + str(EEG_resp.shape[1]) + ' stimulations')
    badchans = pd.read_csv(path_patient_analysis + '/BrainMapping/data/badchan.csv')
    bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])
    # if len(stimlist) != EEG_resp.shape[1]:
    #    print("WARNING: number of stimulations don't agree!")

    lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist,
        lbls)

    bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]
    bad_stims = np.where(labels_region == 'OUT')[0]
    bad_all = np.unique(np.concatenate([bad_region, bad_chans]))
    file_con_trial = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    con_trial_Ph = pd.read_csv(file_con_trial)
    #
    con_trial_Ph = con_trial_Ph[~(np.isin(con_trial_Ph.Chan, bad_all)) & ~(np.isin(con_trial_Ph.Stim, bad_stims))]
    Stims_Ph = np.unique(con_trial_Ph['Stim'])
    con_trial_Ph = con_trial_Ph[np.isin(con_trial_Ph.Stim, Stims_Ph)]
    con_trial_Ph.loc[con_trial_Ph.Sleep == 9, 'Sleep'] = 0
    # con_trial_Ph=con_trial_Ph[con_trial_Ph.Artefact<1]
    con_trial_Ph.loc[con_trial_Ph.P2P > 5000, metric] = np.nan
    con_trial_Ph.loc[con_trial_Ph.LL > 40, metric] = np.nan
    con_trial_Ph.loc[
        con_trial_Ph.Artefact == 1, metric] = np.nan  # con_trial_Ph.loc[con_trial_Ph.Artefact!=0, metric] =np.nan
    # remove outliers
    con_trial_Ph.insert(0, 'zLL', con_trial_Ph.groupby(['Stim', 'Chan', 'Int'])['LL'].transform(
        lambda x: (x - x.mean()) / x.std()).values)
    con_trial_Ph.loc[(con_trial_Ph.zLL > 6), metric] = np.nan
    con_trial_Ph.loc[(con_trial_Ph.zLL < -3), metric] = np.nan

    con_trial_Ph = con_trial_Ph.reset_index(drop=True)
    if not 'SleepState' in con_trial_Ph:
        # con_trial= con_trial[con_trial.d>0]
        con_trial_Ph.insert(5, 'SleepState', 'Wake')
        con_trial_Ph.loc[(con_trial_Ph.Sleep > 1) & (con_trial_Ph.Sleep < 4), 'SleepState'] = 'NREM'
        con_trial_Ph.loc[(con_trial_Ph.Sleep == 4), 'SleepState'] = 'REM'

    sleepstate_labels = np.unique(con_trial_Ph['SleepState'])[::-1]
    # add path
    stimchans = np.unique(con_trial_Ph.Stim).astype('int')
    for i_sc, sc in zip(np.arange(len(stimchans)), stimchans):
        # repeat for each stim channel
        Path(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\NNMF_Stim\\' + labels_all[
            sc] + '\\figures\\').mkdir(
            parents=True, exist_ok=True)
        nmf_fig_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\NNMF_Stim\\' + labels_all[
            sc] + '\\figures\\'
        nmf_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\NNMF_Stim\\' + labels_all[sc] + '/'

        # todo: create function
        V_path = nmf_path + 'IO_' + metric + '.npy'
        title_LL = subj + ', Stim: ' + labels_all[sc] + ', ' + metric + ' as input'
        # todo: remove
        # run_again = 0
        if os.path.isfile(V_path):
            con_trial_nan = con_trial_Ph[
                (con_trial_Ph.Stim == sc) & (con_trial_Ph.d > -1)]  # con_trial_Ph.copy(deep=True)
            NMF_input = np.load(V_path)
        else:
            con_trial_nan = con_trial_Ph[
                (con_trial_Ph.Stim == sc) & (con_trial_Ph.d > -1)]  # con_trial_Ph.copy(deep=True)

            if np.sum(np.isnan(con_trial_nan[metric])) > 0: con_trial_nan[metric] = \
                con_trial_nan.groupby(['Chan', 'Sleep', 'Int'])[metric].transform(
                    lambda x: x.fillna(x.mean()))
            if np.sum(np.isnan(con_trial_nan[metric])) > 0: con_trial_nan[metric] = \
                con_trial_nan.groupby(['Chan', 'Block', 'Int'])[metric].transform(
                    lambda x: x.fillna(x.mean()))
            if np.sum(np.isnan(con_trial_nan[metric])) > 0: con_trial_nan[metric] = \
                con_trial_nan.groupby(['Chan', 'Int'])[metric].transform(
                    lambda x: x.fillna(x.mean()))
            # if np.sum(np.isnan(con_trial_nan[metric])) > 0: con_trial_nan[metric] = con_trial_nan.groupby(['Chan', 'Block'])[
            #     metric].transform(
            # chan_nans = np.unique(con_trial_nan.loc[np.isnan(con_trial_nan[metric]), 'Chan'])
            # if len(chan_nans > 0):
            #     for rc in chan_nans.astype('int'):
            #         mn = np.nanmean(con_trial_nan.loc[(con_trial_nan.Int < 1) & (con_trial_nan.Chan == rc) & (
            #             np.isnan(con_trial_nan[metric])), metric])
            #         st = np.nanstd(con_trial_nan.loc[(con_trial_nan.Int < 1) & (con_trial_nan.Chan == rc) & (
            #             np.isnan(con_trial_nan[metric])), metric])
            #         n = len(
            #             con_trial_nan.loc[(con_trial_nan.Chan == rc) & (np.isnan(con_trial_nan[metric])), metric].values)
            #         con_trial_nan.loc[
            #             (con_trial_nan.Chan == rc) & (np.isnan(con_trial_nan[metric])), metric] = np.random.normal(loc=mn,
            #                                                                                                        scale=st,
            #                                                                                                        size=n)
            # todo:  add random normal distributed value from chan when Int < 1mA (expected value if no response), especially for N peaks

            NMF_input = np.zeros((len(labels_all), len(np.unique(con_trial_nan.Num).astype('int'))))
            i = 0
            nums = con_trial_nan.Num.values
            nums, idx = np.unique(nums, return_index=True)

            for num in nums.astype('int'):
                # for num in nums[np.sort(idx)].astype('int'):
                dat = con_trial_nan[con_trial_nan.Num == num]
                chan = dat.Chan.values.astype('int')
                NMF_input[chan, i] = abs(dat[metric].values)
                i = i + 1
            NMF_input = np.nan_to_num(NMF_input, nan=0)
            np.save(V_path, NMF_input)
            # todo: plot NMF input
            # W
            labels_clean = np.delete(labels_all, bad_all, 0)
            NMF_input_clean = np.delete(NMF_input, bad_all, 0)
            file = nmf_fig_path + 'NMF_input_IO_' + metric
            NMFf.plot_V(NMF_input_clean, subj + ' -- NMF input matrix: ' + metric, ylabels=labels_clean, file=file)

        k0 = 3
        k1 = 7
        num_it = 30
        ranks = np.arange(k0, k1 + 1)
        stab_path = nmf_path + 'stability.npy'
        if os.path.isfile(stab_path):
            stability, instability = np.load(stab_path)
        else:
            stability, instability = NMFf.get_stability(NMF_input, num_it=num_it, k0=k0, k1=k1)
            np.save(stab_path, [stability, instability])
            title_stab = subj + ' -- IO CR -- Stability NNMF, iterations: ' + str(num_it)
            NMFf.plot_stability(stability, instability, k0, k1, title_stab, nmf_fig_path)
        # select rank
        stab_sel = (stability / stability.max() - instability / instability.max())
        ix0 = np.argmax(stab_sel)
        stab_sel[ix0] = 0
        ix1 = np.argmax(stab_sel)
        rank_sel = np.stack([ranks[ix0], ranks[ix1]])
        # todo: remove later, this is only to rerun for specific ranks

        p = 1
        for rk in [ranks[ix0]]:  # rank_sel:  # range(k0, k1 + 1):
            if rk == 2: rk = 3
            [W, W0, H] = NMFf.get_nnmf_Epi(NMF_input, rk, it=4000)
            # columns label
            col0 = ['Stim', 'Int', 'Block', 'Hour', 'Date', 'Sleep']  #  'Hour',
            col = ['Stim', 'Int', 'Block', 'Hour', 'Date', 'Sleep']  #  'Hour',
            W_col = []
            H_col = []
            for i in range(H.shape[0]):
                col.append('H' + str(i + 1))
                W_col.append('W' + str(i + 1))
                H_col.append('H' + str(i + 1))

            con_nmf = np.zeros((H.shape[1], len(col0) + H.shape[0]))
            con_nmf[:, len(col0):] = H.T
            # add stim channel, Hour and Intensity
            summ = con_trial_nan.groupby(['Num'], as_index=False)[col0].mean()
            con_nmf[:, 0:len(col0)] = summ.values[:, 1:]
            con_nmf = pd.DataFrame(con_nmf, columns=col)
            # sleepstate
            con_nmf.insert(5, 'SleepState', 'Wake')
            con_nmf.loc[(con_nmf.Sleep > 1) & (con_nmf.Sleep < 4), 'SleepState'] = 'NREM'
            con_nmf.loc[(con_nmf.Sleep == 4), 'SleepState'] = 'REM'

            con_nmf.insert(0, 'Stim_L', labels_all[sc])
            con_nmf.insert(0, 'Area', labels_region[sc])
            # todo: plot all H against Int
            NMFf.plot_H_trial(con_nmf, 'Int', 'Stim_L', title_LL, nmf_fig_path)
            # NMFf.plot_H_trial(pd_con_nnmf, 'Int', 'Block', title_LL, nmf_fig_path)
            # W
            labels_clean = np.delete(labels_all, bad_all, 0)
            W_clean = np.delete(W, bad_all, 0)
            file = nmf_fig_path + 'W_r' + str(rk)
            title = subj + ' -- Basic Function, Rank: ' + str(rk) + '  -- ' + cond_folder
            NMFf.plot_W(W_clean, title, labels_clean, file)

            if np.isin(rk, rank_sel):
                # NMFf.plot_H_trial(con_nmf, 'Int', 'Block', title_LL, nmf_fig_path)
                file = nmf_path + 'IO_CR_' + str(p) + 'rk' + str(rk) + '.csv'
                con_nmf.to_csv(file, index=False, header=True)

                # associate H to stim channel
                NNMF_ass = NMFf.get_NMF_Stim_association(con_nmf, H_col)
                for cond in ['Block', 'Sleep', 'SleepState']:
                    NNMF_AUC = NMFf.get_NMF_AUC(con_nmf, NNMF_ass, cond_sel=cond)
                    NNMF_AUC.insert(0, 'Stim_L', labels_all[sc])
                    NNMF_AUC.insert(0, 'Area', labels_region[sc])

                    file = nmf_path + 'IO_' + cond + '_AUC_' + str(p) + 'rk' + str(rk) + '.csv'
                    NNMF_AUC.to_csv(file, index=False, header=True)
                file_ass = nmf_path + 'IO_association_' + str(p) + 'rk' + str(rk) + '.csv'
                NNMF_ass.to_csv(file_ass, index=False, header=True)
                # store basic function (W) values
                W_save = pd.DataFrame(W, columns=W_col)
                W_save.insert(0, 'Label', labels_all)
                file = nmf_path + 'W_' + str(p) + 'rk' + str(rk) + '.csv'
                W_save.to_csv(file, index=False, header=True)
                # plot AUC¬
                # ssave H
                file = nmf_fig_path + 'NMF_H_rk' + str(rk)
                NMFf.plot_H(H, subj + ' -- Activation Function H ', file=file)
                # NNMF_AUC = NNMF_AUC[NNMF_AUC.Pearson > -1]
                for sc in np.unique(NNMF_ass.Stim).astype('int'):
                    for H in np.unique(NNMF_ass.loc[NNMF_ass.Stim == sc, 'H_num']).astype('int'):
                        file = nmf_fig_path + 'IO_Sleep_AUC_rk' + str(rk) + '_H' + str(H) + '_Stim_' + labels_all[
                            sc]  # +'.npy'
                        title = subj + ' -- Sleep -- ' + labels_all[sc] + ', H' + str(H) + '/' + str(rk)
                        NMFf.plot_NMF_AUC_Sleep(con_nmf, sc, H, title, file)
                        title = subj + ' -- SleepStat -- ' + labels_all[sc] + ', H' + str(H) + '/' + str(rk)
                        file = nmf_fig_path + 'IO_SleepState_AUC_rk' + str(rk) + '_H' + str(H) + '_Stim_' + labels_all[
                            sc]  # +'.npy'
                        NMFf.plot_NMF_AUC_SleepState(con_nmf, sc, H, title, file)
                p = p + 1
    print(subj + ' ---- DONE ------ ')


# compute_subj('EL013')
# compute_subj('EL011')

print('START')
metrics = ['LL']  # 'sN2','sN1',
for subj in ["EL019", "EL017",
             "EL018"]:  # ["EL016", "EL011", "EL004", "EL005", "EL010",  "EL015", "El014"]:  # ["EL011","EL015", "EL010",  "EL012", "El014"]: #, "EL004", "EL010", "EL011", "EL012", "El014"]:  # "EL012", "EL013",
    for m in metrics:
        compute_subj(subj, m)
#         try:
#             _thread.start_new_thread(compute_subj, (subj,m))
#         except e:
#             print('error')
# while 1:
#     time.sleep(1)
