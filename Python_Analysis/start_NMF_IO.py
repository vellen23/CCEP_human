import sys

sys.path.append('./py_functions')
sys.path.append('./PCI/')
sys.path.append('./PCI/PCIst')
import os
import numpy as np
import pandas as pd
from tkinter import filedialog
from tkinter import *
import matplotlib.pyplot as plt
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
folder ='InputOutput'
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
if platform.system() == 'Windows':
    # sep = ','
    path = 'T:\EL_experiment\Patients\\'  # + subj
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
cond_folder = 'Ph'
##subjs = ['EL003''EL004', 'EL005', ]  # EL003',
subjs = ['El003','EL008', 'EL010', 'EL012', 'EL014']
for subj in subjs:
    path_patient = os.path.join(path, subj)
    # only used for reading labels
    files_list = glob(path_patient + '/Analysis/InputOutput/data/Stim_list_*')
    i = 0
    stimlist = pd.read_csv(files_list[i])

    badchans = pd.read_csv(path_patient + '/Analysis/InputOutput/data/badchan.csv')
    bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])

    lbls = pd.read_excel(path_patient + "/infos/" + subj + "_labels.xlsx", header=0, sheet_name='BP')
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist,
        lbls)

    bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]
    bad_stims = np.where(labels_region == 'OUT')[0]
    file_con = path_patient + '/Analysis/' + folder + '/' + cond_folder + '/data/con_trial.csv'
    con_trial_Ph = pd.read_csv(file_con)
    con_trial_Ph.loc[np.isnan(con_trial_Ph.LL), 'LL_peak'] = np.nan

    #
    con_trial_Ph = con_trial_Ph[
        (np.isin(con_trial_Ph.Condition, [1, 3])) & ~(np.isin(con_trial_Ph.Chan, bad_region)) & ~(
            np.isin(con_trial_Ph.Stim, bad_region))]
    Stims_Ph = np.unique(con_trial_Ph.loc[con_trial_Ph.Condition == 3, 'Stim'])
    con_trial_Ph = con_trial_Ph[np.isin(con_trial_Ph.Stim, Stims_Ph)]

    # add path
    nmf_fig_path = path_patient + '/Analysis/InputOutput/' + cond_folder + '/NNMF/figures/'
    nmf_path = path_patient + '/Analysis/InputOutput/' + cond_folder + '/NNMF/'
    for path_find in [nmf_path, nmf_fig_path]:
        try:
            os.mkdir(path_find)
        except OSError:
            print(path_find + " -- already exists")

    # todo: create function

    # todo: create function
    V_path = nmf_path + 'IO_LLpeak.npy'
    title_LL = subj + ', Stim: all, LL as input'
    # todo: remove
    run_again = 0
    # if run_again:
    if os.path.isfile(V_path):
        NMF_input = np.load(V_path)
    else:
        con_trial_nan = con_trial_Ph.copy(deep=True)
        con_trial_nan = con_trial_nan[(con_trial_nan.d > 7)]
        title_LL = subj + ', Stim: all, LL as input'
        con_trial_nan.LL_peak = con_trial_nan.groupby(['Stim', 'Chan', 'Int', 'Condition'])['LL_peak'].transform(
            lambda x: x.fillna(x.mean()))
        con_trial_nan.LL_peak = con_trial_nan.groupby(['Stim', 'Chan', 'Int'])['LL_peak'].transform(
            lambda x: x.fillna(x.mean()))
        con_trial_nan.LL_peak = con_trial_nan.groupby(['Stim', 'Chan'])['LL_peak'].transform(lambda x: x.fillna(x.mean()))
        con_trial_nan.LL_peak = con_trial_nan.groupby(['Chan'])['LL_peak'].transform(lambda x: x.fillna(x.mean()))

        NMF_input = np.zeros((len(labels_all), len(np.unique(con_trial_nan.Num).astype('int'))))
        i = 0
        nums = con_trial_nan.Num.values
        nums, idx = np.unique(nums, return_index=True)

        for num in nums.astype('int'):
            # for num in nums[np.sort(idx)].astype('int'):
            dat = con_trial_nan[con_trial_nan.Num == num]
            chan = dat.Chan.values.astype('int')
            NMF_input[chan, i] = abs(dat.LL_peak.values)
            i = i + 1
        NMF_input = np.nan_to_num(NMF_input, nan=0)
        np.save(V_path, NMF_input)
        # W
        labels_clean = np.delete(labels_all, bad_region, 0)
        NMF_input_clean = np.delete(NMF_input, bad_region, 0)
        file = nmf_fig_path + 'NMF_input_IO_LLpeak'
        NMFf.plot_V(NMF_input_clean, subj + ' -- NMF input matrix: LL ', ylabels=labels_clean, file=file)

    k0 = 3
    k1 = 8
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

    p =1
    for rk in rank_sel:  # range(k0, k1 + 1):
        [W, H] = NMFf.get_nnmf(NMF_input, rk, it=2000)
        # columns label
        col0 = ['Stim', 'Int', 'Condition', 'Hour', 'Date', 'Sleep']  #  'Hour',
        col = ['Stim', 'Int', 'Condition', 'Hour', 'Date', 'Sleep']  #  'Hour',
        W_col = []
        H_col =[]
        for i in range(H.shape[0]):
            col.append('H' + str(i + 1))
            W_col.append('W' + str(i + 1))
            H_col.append('H' + str(i + 1))
        con_nmf = np.zeros((H.shape[1], len(col0) + H.shape[0]))
        con_nmf[:, len(col0):] = H.T
        # add stim channel, Hour and Intensity
        summ = con_trial_Ph.groupby(['Num'], as_index=False)[col0].mean()
        con_nmf[:, 0:len(col0)] = summ.values[:, 1:]
        con_nmf = pd.DataFrame(con_nmf, columns=col)

        con_nmf.insert(0, 'Stim_L', 0)
        con_nmf.insert(0, 'Area', 0)
        for scc in Stims_Ph.astype(int):
            con_nmf.loc[con_nmf.Stim == scc, 'Stim_L'] = labels_all[scc]
            con_nmf.loc[con_nmf.Stim == scc, 'Area'] = labels_region[scc]

        NMFf.plot_H_trial(con_nmf, 'Int', 'Stim_L', title_LL, nmf_fig_path)
        NMFf.plot_H_trial(con_nmf, 'Int', 'Condition', title_LL, nmf_fig_path)
        # W
        labels_clean = np.delete(labels_all, bad_region, 0)
        W_clean = np.delete(W, bad_region, 0)
        file = nmf_fig_path + 'W_r' + str(rk)
        title = subj + ' -- Basic Function, Rank: ' + str(rk) + '  -- ' + cond_folder
        NMFf.plot_W(W_clean, title, labels_clean, file)

        if np.isin(rk, rank_sel):
            file = nmf_path + 'IO_Ph_LLpeak_'+str(p)+'rk' + str(rk) + '.npy'
            con_nmf.to_csv(file, index=False, header=True)

            # associate H to stim channel
            NNMF_ass = NMFf.get_NMF_Stim_association(con_nmf, H_col)
            NNMF_AUC = NMFf.get_NMF_AUC(con_nmf, NNMF_ass, cond_sel='Condition')
            NNMF_AUC.insert(0, 'Stim_L', 0)
            NNMF_AUC.insert(0, 'Area', 0)
            NNMF_AUC.insert(0, 'Cond_Label', 0)
            for sc in Stims_Ph.astype(int):
                NNMF_AUC.loc[NNMF_AUC.Stim == sc, 'Stim_L'] = labels_all[sc]
                NNMF_AUC.loc[NNMF_AUC.Stim == sc, 'Area'] = labels_region[sc]
            for sc in np.unique(NNMF_AUC.Condition).astype(int):
                NNMF_AUC.loc[NNMF_AUC.Condition == sc, 'Cond_Label'] = cond_labels[sc]

            file = nmf_path + 'IO_Ph_AUC_LLpeak_'+str(p)+'rk' + str(rk) + '.csv'
            NNMF_AUC.to_csv(file, index=False, header=True)
            file_ass = nmf_path + 'IO_association_' + str(p) + 'rk' + str(rk) + '.csv'
            NNMF_ass.to_csv(file_ass, index=False, header=True)
            # store basic function (W) values
            W_save = pd.DataFrame(W, columns=W_col)
            W_save.insert(0, 'Label', labels_all)
            file = nmf_path + 'W_LLpeak_'+str(p)+'rk' + str(rk) + '.csv'
            W_save.to_csv(file, index=False, header=True)
            # plot AUC¬
            # #ssave H
            # fig = plt.figure(figsize=(H.shape[1]/10,H.shape[0]))
            # plt.imshow(H, aspect=20, vmin=np.percentile(H, 50),vmax=np.percentile(H, 95))# , vmin=0, vmax=15
            # plt.ylabel('Channels')
            # plt.yticks(np.arange(H.shape[0]), H_col)
            # plt.xlabel('trials')
            # plt.title(subj+' -- Activation Function H ')
            # file = nmf_fig_path + 'NMF_H_rk'+ str(rk)
            # plt.colorbar()
            # plt.savefig(file + '.svg')
            # plt.savefig(file + '.jpg')
            # plt.show()

            for sc in np.unique(NNMF_ass.Stim).astype('int'):
                for H in np.unique(NNMF_ass.loc[NNMF_ass.Stim == sc, 'H_num']).astype('int'):
                    file = nmf_fig_path + 'IO_Ph_AUC_LLpeak_rk' + str(rk) + '_H' + str(H) + '_Stim_' + labels_all[
                        sc]  # +'.npy'
                    title = subj + ' -- ' + labels_all[sc] + ', H' + str(H) + '/' + str(rk)
                    NMFf.plot_NMF_AUC_Ph(con_nmf, sc, H, title, file)
            p = p+1