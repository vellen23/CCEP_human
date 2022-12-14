import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import seaborn as sns
import pandas as pd
import sys

sys.path.append('T:\EL_experiment\Codes\CCEP_human\Python_Analysis\py_functions')
from scipy.stats import norm
from tkinter import *
import _thread

root = Tk()
root.withdraw()
import scipy
import NMF_funcs as NMFf
import basic_func as bf
import tqdm
from matplotlib.patches import Rectangle
import freq_funcs as ff
# from tqdm.notebook import trange, tqdm
# remove some warnings
import warnings
from pathlib import Path
import LL_funcs as LLf
import significance_funcs as sigf
import copy

# I expect to see RuntimeWarnings in this block
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sleepstate_labels = ['NREM', 'REM', 'Wake']

folder = 'BrainMapping'
cond_folder = 'CR'


class main:
    def __init__(self, subj):
        #  basics, get 4s of data for each stimulation, [-2,2]s
        self.folder = 'BrainMapping'
        self.cond_folder = 'CR'
        self.path_patient_analysis = 'y:\\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
        path_gen = os.path.join('y:\\eLab\Patients\\' + subj)
        if not os.path.exists(path_gen):
            path_gen = 'T:\\EL_experiment\\Patients\\' + subj
        self.path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
        path_infos = os.path.join(self.path_patient, 'infos')
        if not os.path.exists(path_infos):
            path_infos = path_gen + '\\infos'

        self.Fs = 500
        self.dur = np.zeros((1, 2), dtype=np.int32)
        self.dur[0, :] = [-1, 3]
        self.dur_tot = np.int32(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        # load patient specific information
        lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
        self.labels_all = lbls.label.values
        self.labels_C = lbls.Clinic.values

        stimlist = pd.read_csv(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\stimlist_' + self.cond_folder + '.csv')
        if len(stimlist) == 0:
            stimlist = pd.read_csv(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\stimlist_' + self.cond_folder + '.csv')
        #
        labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
            stimlist,
            lbls)
        bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]
        self.labels_region_L = lbls.Hemisphere.values + '_' + labels_region
        self.subj = subj
        self.labels_region = labels_region

        # regions information
        self.CR_color = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\Analysis\BrainMapping\CR_color.xlsx",
                                      header=0)
        regions = pd.read_excel("Y:\eLab\EvM\Projects\EL_experiment\Analysis\Patients\Across\elab_labels.xlsx",
                                sheet_name='regions',
                                header=0)
        self.color_regions = regions.color.values
        self.regions = regions
        badchans = pd.read_csv(self.path_patient_analysis + '/BrainMapping/data/badchan.csv')
        self.bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])
        # C = regions.label.values
        # self.path_patient   = path_patient
        # self.path_patient_analysis = os.path.join(os.path.dirname(os.path.dirname(self.path_patient)), 'Projects\EL_experiment\Analysis\Patients', subj)
        ##bad channels
        non_stim = np.arange(len(self.labels_all))
        non_stim = np.delete(non_stim, StimChanIx, 0)
        WM_chans = np.where(self.labels_region == 'WM')[0]
        self.bad_all = np.unique(np.concatenate([WM_chans, bad_region, self.bad_chans, non_stim])).astype('int')
        stim_chans = np.arange(len(labels_all))
        self.stim_chans = np.delete(stim_chans, self.bad_all, 0)
        Path(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures').mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\NNMF').mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\NNMF\\figures').mkdir(
            parents=True, exist_ok=True)
        for group in ['\\General', '\\Block', '\\Sleep']:
            for metric in ['\\BM_LL', '\\BM_Prob', '\\BM_binary', '\\BM_sym', '\\BM_change', '\\BM_Dir']:
                Path(
                    self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures' + group + metric).mkdir(
                    parents=True, exist_ok=True)

        # labels:
        labels_sel = np.delete(self.labels_all, self.bad_all, 0)
        areas_sel = np.delete(self.labels_region_L, self.bad_all, 0)
        # sort
        ind = np.argsort(areas_sel)
        areas_sel = np.delete(self.labels_region, self.bad_all, 0)
        self.labels_sel = labels_sel[ind]
        self.areas_sel = areas_sel[ind]
        self.ind = ind

        ## some important data

    def plot_BM(self, M, labels, areas, label, t, method, save=1, circ=1, cmap='hot', group='block'):
        cmap_sym = ListedColormap(["#2b2b2b", "black", "#c34f2f", "#1e4e79"])
        cmap_binary = ListedColormap(["#2b2b2b", "black", "#bdd7ee"])
        cmap_LL = 'hot'
        cmap_Prob = 'afmhot'
        cmap_change = ListedColormap(
            ["#2b2b2b", "black", "white", "blue", "red"])  # ListedColormap(["blue", "white", "red"])
        cmap_Dir = 'bwr'

        cmap = locals()["cmap_" + method]

        M[np.isnan(M)] = -1

        fig = plt.figure(figsize=(15, 15))
        fig.patch.set_facecolor('xkcd:white')
        axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])  # x, y, (start posiion), lenx, leny
        if method == 'LL':
            cmap = copy.copy(plt.cm.get_cmap(cmap))  # cmap = copy.copy(mpl.cm.get_cmap(cmap))
            cmap.set_under('#2b2b2b')
            cmap.set_bad('black')
            M = np.ma.masked_equal(M, 0)
            im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=np.percentile(M, 95))
        elif method == 'Prob':
            cmap = copy.copy(plt.cm.get_cmap(cmap))  # cmap = copy.copy(mpl.cm.get_cmap(cmap))
            cmap.set_under('black')
            cmap.set_bad('#2b2b2b')
            M = np.ma.masked_equal(M, -1)
            im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=0.1, vmax=0.95)
        elif method == 'Dir':
            cmap = copy.copy(plt.cm.get_cmap(cmap))  # cmap = copy.copy(mpl.cm.get_cmap(cmap))
            cmap.set_under('black')
            cmap.set_bad('white')
            M = np.ma.masked_equal(M, -20)
            mask = 1 * (np.tri(M.shape[0], k=0) == 0)
            M = np.ma.array(M, mask=mask)
            im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=-1, vmax=1)
        else:
            im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap=cmap)
        plt.suptitle(label + '-- ' + method)
        plt.xlim([-1.5, len(labels) - 0.5])
        plt.ylim([-0.5, len(labels) + 0.5])
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            r = areas[i]
            axmatrix.add_patch(Rectangle((i - 0.5, len(labels) - 0.5), 1, 1, alpha=1,
                                         facecolor=self.color_regions[np.where(self.regions == r)[0][0]]))
            axmatrix.add_patch(
                Rectangle((-1.5, i - 0.5), 1, 1, alpha=1,
                          facecolor=self.color_regions[np.where(self.regions == r)[0][0]]))
        # Plot colorbar.

        if circ:
            time = str(t).zfill(2) + ':00'
            axcolor = fig.add_axes([0.04, 0.85, 0.08, 0.08])  # x, y, x_len, y_len
            circle1 = plt.Circle((0.5, 0.5), 0.4, color=self.CR_color.c[t], alpha=self.CR_color.a[t])
            plt.text(0.3, 0.3, time)
            plt.axis('off')
            axcolor.add_patch(circle1)
        axcolor = fig.add_axes([0.9, 0.15, 0.01, 0.7])  # x, y, x_len, y_len
        plt.colorbar(im, cax=axcolor)

        # plt.savefig(path_patient + '/Analysis/BrainMapping/CR/figures/BM_plot/BM_'+label+'.svg')

        if save:
            plt.savefig(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\' + group + '\\BM_' + method + '\\BM_' + label + '.jpg')
            plt.savefig(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\' + group + '\\BM_' + method + '\\BM_' + label + '.svg')

            # plt.savefig(path_patient_analysis+'\\' + folder + '\\' + cond_folder +'\\figures/BM_LL\\BM_'+label+'.jpg')
            plt.close(fig)  # plt.show()#

        else:
            plt.show()

    def get_sig(self, sc, rc, con_trial, M_GT, t_resp, EEG_CR, p=95, exp=2, w_cluster=0.25, t_0=1):
        # for each trial get significance level based on surrogate (Pearson^2 * LL)
        dat = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1)]
        EEG_trials = ff.lp_filter(EEG_CR[[[rc]], dat.Num.values.astype('int'), :], 45, self.Fs)
        LL_trials = LLf.get_LL_all(EEG_trials, self.Fs, w_cluster)
        # surr
        pear_surr_all = []
        f = 1
        for t_test in [0.3, 0.7, 1.8, 2.2, 2.6]:  # surrogates times, todo: in future blockwise
            s = (-1) ** f
            pear = np.zeros((len(EEG_trials[0]),)) - 1
            for n_c in range(len(M_GT)):
                pear = np.max([pear, sigf.get_pearson2mean(M_GT[n_c, :], s * EEG_trials[0], tx=t_0 + t_resp, ty=t_test,
                                                           win=w_cluster,
                                                           Fs=500)], 0)
            LL = LL_trials[0, :, int((t_test + w_cluster / 2) * self.Fs)]
            # pear_surr = np.arctanh(np.max([pear,pear2],0))*LL
            pear_surr = np.sign(pear) * abs(pear ** exp) * LL
            pear_surr_all = np.concatenate([pear_surr_all, pear_surr])
            f = f + 1
        # other trials
        real_trials = np.unique(
            con_trial.loc[(con_trial.Stim == sc) & (con_trial.Chan == rc), 'Num'].values.astype('int'))
        stim_trials = np.unique(
            con_trial.loc[(con_trial.Stim >= rc - 1) & (con_trial.Stim <= rc + 1), 'Num'].values.astype('int'))
        StimNum = np.random.choice(np.unique(con_trial.Num), size=400)
        StimNum = [i for i in StimNum if i not in stim_trials]
        StimNum = [i for i in StimNum if i not in stim_trials + 1]
        StimNum = [i for i in StimNum if i not in real_trials]

        StimNum = np.unique(StimNum).astype('int')
        EEG_surr = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 45, self.Fs)
        bad_StimNum = np.where(np.max(abs(EEG_surr[0]), 1) > 1000)
        if (len(bad_StimNum[0]) > 0):
            StimNum = np.delete(StimNum, bad_StimNum)
            EEG_surr = ff.lp_filter(EEG_CR[[[rc]], StimNum, :], 45, self.Fs)
        LL_surr = LLf.get_LL_all(EEG_surr, self.Fs, w_cluster)
        for t_test in [0.3, 0.7, 1.8, 2.2, 2.6]:  # surrogates times, todo: in future blockwise
            s = (-1) ** f

            pear = np.zeros((len(EEG_surr[0]),)) - 1
            for n_c in range(len(M_GT)):
                pear = np.max([pear, sigf.get_pearson2mean(M_GT[n_c, :], s * EEG_surr[0], tx=t_0 + t_resp, ty=t_test,
                                                           win=w_cluster,
                                                           Fs=500)], 0)

            LL = LL_surr[0, :, int((t_test + w_cluster / 2) * self.Fs)]
            # pear_surr = np.arctanh(np.max([pear,pear2],0))*LL
            pear_surr = np.sign(pear) * abs(pear ** exp) * LL
            pear_surr_all = np.concatenate([pear_surr_all, pear_surr])
            f = f + 1

        # real
        t_test = t_0 + t_resp
        pear = np.zeros((len(EEG_trials[0]),)) - 1
        for n_c in range(len(M_GT)):
            pear = np.max(
                [pear, sigf.get_pearson2mean(M_GT[n_c, :], s * EEG_trials[0], tx=t_0 + t_resp, ty=t_test, win=w_cluster,
                                             Fs=500)], 0)

        LL = LL_trials[0, :, int((t_test + w_cluster / 2) * self.Fs)]
        pear = np.sign(pear) * abs(pear ** exp) * LL
        sig = (pear > np.nanpercentile(pear_surr_all, p)) * 1
        con_trial.loc[
            (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1), 'Sig'] = sig  # * sig_mean
        con_trial.loc[
            (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1), 'LL_onset'] = LL
        return con_trial

    def save_M_block(self, con_trial, metrics=['LL'], savefig=1):
        con_trial_sig = con_trial[(con_trial.Sleep < 5) & (con_trial.d > -10)]
        con_trial_sig.loc[con_trial_sig.Sig < 0, 'Sig'] = np.nan
        con_trial_sig.insert(4, 'LL_sig', np.nan)
        con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL_sig'] = con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL']
        con_trial_sig = con_trial_sig.drop(columns='LL')
        con_trial_sig.insert(4, 'LL', con_trial_sig.LL_sig)
        con_trial_sig.insert(4, 'Prob', con_trial_sig.Sig)
        con_trial_sig = con_trial_sig.drop(columns='LL_sig')
        # labels:
        labels_sel = np.delete(self.labels_all, self.bad_all, 0)
        areas_sel = np.delete(self.labels_region_L, self.bad_all, 0)
        # sort
        ind = np.argsort(areas_sel)
        areas_sel = np.delete(self.labels_region, self.bad_all, 0)
        labels_sel = labels_sel[ind]
        areas_sel = areas_sel[ind]
        # blocks
        for metric in metrics:
            M_dir_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Block\\BM_' + metric + '.npy'
            if os.path.exists(M_dir_path):
                M_B_all = np.load(M_dir_path)
            else:

                M_B_all = np.zeros((int(np.max(con_trial_sig.Block) + 1), len(self.labels_all), len(self.labels_all)))
                for b in np.unique(con_trial_sig.Block).astype('int'):
                    summ = con_trial_sig[
                        (con_trial_sig.Block == b) & (con_trial_sig.Artefact < 1)]  # (con_trial.Sleep==s)&

                    summ = summ.groupby(['Stim', 'Chan'], as_index=False)[[metric, 'd']].mean()
                    # summ = summ[summ.Sig==1]
                    t = np.bincount(con_trial_sig.loc[con_trial_sig.Block == b, 'Hour']).argmax()
                    M = np.zeros((len(self.labels_all), len(self.labels_all))) - 1
                    # M[:, :] = np.nan
                    for sc in np.unique(summ.Stim).astype('int'):
                        chan = summ.loc[summ.Stim == sc, 'Chan'].values.astype('int')
                        LL = summ.loc[summ.Stim == sc, metric].values
                        M[sc, chan] = LL
                    # M[np.isnan(M)] = 0
                    M_B_all[b, :, :] = M
                    if savefig:
                        M_resp = np.delete(np.delete(M, self.bad_all, 0), self.bad_all, 1)
                        M_resp = M_resp[ind, :]
                        M_resp = M_resp[:, ind]
                        ll = 'block' + str(int(b)).zfill(2)  # +', '+sleep_states[s]
                        self.plot_BM(M_resp, labels_sel, areas_sel, ll, t, metric, save=1, circ=1, group='Block')
                        # self.plot_BM_CR_block(M_resp, labels_sel, areas_sel, ll, t, metric, savefig)
                np.save(M_dir_path, M_B_all)
            # pearson correlation across all blocks
            M_B_allp = np.zeros((int(np.max(con_trial.Block) + 1), len(self.labels_all), len(self.labels_all)))
            M_B_allp[:, :, :] = M_B_all[:, :, :]
            M_B_allp[np.isnan(M_B_allp)] = 0
            M_B_pear = np.zeros((int(np.max(con_trial.Block) + 1), int(np.max(con_trial.Block) + 1)))
            M_pearson = np.zeros((1, 9))
            block_all = np.unique(con_trial.Block).astype('int')
            for b1 in block_all:
                for b2 in block_all[block_all > b1]:
                    M_B_pear[b1, b2] = np.corrcoef(M_B_allp[b1, :, :].flatten(), M_B_allp[b2, :, :].flatten())[0, 1]
                    M_B_pear[b2, b1] = np.corrcoef(M_B_allp[b1, :, :].flatten(), M_B_allp[b2, :, :].flatten())[0, 1]
                    # quantification
                    M_pearson_c = np.zeros((1, 9))
                    s1 = np.bincount(con_trial.loc[con_trial.Block == b1, 'Sleep']).argmax()
                    h1 = np.bincount(con_trial.loc[con_trial.Block == b1, 'Hour']).argmax()
                    m1 = len(np.unique(con_trial.loc[con_trial.Block == b1, 'SleepState']))
                    s2 = np.bincount(con_trial.loc[con_trial.Block == b2, 'Sleep']).argmax()
                    h2 = np.bincount(con_trial.loc[con_trial.Block == b2, 'Hour']).argmax()
                    m2 = len(np.unique(con_trial.loc[con_trial.Block == b2, 'SleepState']))

                    M_pearson_c[0] = [b1, b2, M_B_pear[b1, b2], s1, s2, h1, h2, m1, m2]
                    M_pearson = np.concatenate([M_pearson, M_pearson_c], 0)
            M_pearson = M_pearson[1:, :]
            M_pear = pd.DataFrame(M_pearson,
                                  columns=['A_Block', 'B_Block', 'Pearson', 'A_Sleep', 'B_Sleep', 'A_Hour', 'B_Hour',
                                           'A_mix', 'B_mix'])
            M_pear.loc[M_pear.A_Sleep == 2, 'A_Sleep'] = 3
            M_pear.loc[M_pear.B_Sleep == 2, 'B_Sleep'] = 3
            M_pear.insert(0, 'Comp_Type', 'W-W')
            M_pear.loc[(M_pear.B_Sleep == 3) & (M_pear.A_Sleep == 3), 'Comp_Type'] = 'NREM-NREM'
            M_pear.loc[(M_pear.B_Sleep == 4) & (M_pear.A_Sleep == 4), 'Comp_Type'] = 'REM-REM'

            M_pear.loc[(M_pear.B_Sleep == 0) & (M_pear.A_Sleep == 4), 'Comp_Type'] = 'W-REM'
            M_pear.loc[(M_pear.B_Sleep == 4) & (M_pear.A_Sleep == 0), 'Comp_Type'] = 'W-REM'

            M_pear.loc[(M_pear.B_Sleep == 0) & (M_pear.A_Sleep == 3), 'Comp_Type'] = 'W-NREM'
            M_pear.loc[(M_pear.B_Sleep == 3) & (M_pear.A_Sleep == 0), 'Comp_Type'] = 'W-NREM'

            M_pear.loc[(M_pear.B_Sleep == 4) & (M_pear.A_Sleep == 3), 'Comp_Type'] = 'REM-NREM'
            M_pear.loc[(M_pear.B_Sleep == 3) & (M_pear.A_Sleep == 4), 'Comp_Type'] = 'REM-NREM'
            ### figures
            # 1. Quantification
            sns.catplot(x='Comp_Type', y='Pearson', kind='box', data=M_pear[(M_pear.A_mix == 1) & (M_pear.B_mix == 1)],
                        height=8, aspect=3)
            plt.title('Pearson of connectivity maps comparisons (without blocks having mixed sleepstates)', fontsize=30)
            plt.ylabel('Pearson Correlation', fontsize=25)
            plt.xlabel('Comparison Type', fontsize=25)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ylim([0.7, 1])
            plt.savefig(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Block\\BM_' + metric + '\\BM_corr_quant.jpg')
            plt.savefig(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Block\\BM_' + metric + '\\BM_corr_quant.svg')

            # Pearson plot
            fig = plt.figure(figsize=(8, 8))
            M_B_pear[M_B_pear < 0.5] = np.nan
            cmap = copy.copy(plt.cm.get_cmap('hot'))
            cmap.set_under('#2b2b2b')
            cmap.set_bad('#2b2b2b')
            M_B_pear = np.ma.masked_equal(M_B_pear, 0)
            M_B_pear = np.ma.masked_equal(M_B_pear, np.nan)
            plt.pcolor(M_B_pear, cmap='hot', vmin=np.max([0.8, np.min([0.9, np.nanpercentile(M_B_pear, 10)])]),
                       vmax=np.min([0.95, np.nanpercentile(M_B_pear, 95)]))
            # self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder +'\\BM_figures\\Block\\BM_'+metric+'.npy'
            plt.savefig(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Block\\BM_' + metric + '\\BM_corr.jpg')
            plt.savefig(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Block\\BM_' + metric + '\\BM_corr.svg')
            plt.close(fig)  # plt.show()#

        return M_B_all

    def save_M_sleep(self, con_trial, metrics=['LL'], savefig=1):
        # 1. convert con_trial table
        con_trial_sig = con_trial[con_trial.d > -10]
        # con_trial_sig.loc[con_trial_sig.Sig == 1, 'Sig'] = 0
        # con_trial_sig.loc[con_trial_sig.Sig == 2, 'Sig'] = 1
        con_trial_sig.loc[con_trial_sig.Sig < 0, 'Sig'] = np.nan
        con_trial_sig.insert(4, 'LL_sig', np.nan)
        con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL_sig'] = con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL']
        con_trial_sig = con_trial_sig.drop(columns='LL')
        con_trial_sig.insert(4, 'LL', con_trial_sig.LL_sig)
        con_trial_sig.insert(4, 'Prob', con_trial_sig.Sig)
        con_trial_sig = con_trial_sig.drop(columns='LL_sig')

        sleepstate_labels = np.unique(con_trial_sig['SleepState'])

        # blocks
        for metric in metrics:
            M_dir_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Sleep\\BM_' + metric + '.npy'
            M_B_all = np.zeros((len(sleepstate_labels), len(self.labels_all), len(self.labels_all)))
            for ss, i in zip(sleepstate_labels, np.arange(len(sleepstate_labels))):
                summ = con_trial_sig[
                    (con_trial_sig.SleepState == ss) & (con_trial_sig.Artefact < 1)]  # (con_trial.Sleep==s)&
                summ = summ.groupby(['Stim', 'Chan'], as_index=False)[[metric, 'd']].mean()
                # summ = summ[summ.Sig==1]

                M = np.zeros((len(self.labels_all), len(self.labels_all))) - 1
                # M[:, :] = np.nan
                for sc in np.unique(summ.Stim).astype('int'):
                    chan = summ.loc[summ.Stim == sc, 'Chan'].values.astype('int')
                    LL = summ.loc[summ.Stim == sc, metric].values
                    M[sc, chan] = LL
                M[np.isnan(M)] = -1
                M_B_all[i, :, :] = M
                if savefig:
                    M_resp = np.delete(np.delete(M, self.bad_all, 0), self.bad_all, 1)
                    M_resp = M_resp[self.ind, :]
                    M_resp = M_resp[:, self.ind]
                    self.plot_BM(M_resp, self.labels_sel, self.areas_sel, ss, i, metric, save=1, circ=0, group='Sleep')
                    # self.plot_BM_CR_block(M_resp, labels_sel, areas_sel, ll, t, metric, savefig)
            np.save(M_dir_path, M_B_all)

    def get_sleep_ttest(self, con_trial, load=1):
        sleepstate_labels_u = np.unique(con_trial.loc[con_trial.Sleep > 0, 'SleepState'])
        sleepstate_labels = ['NREM', 'REM']
        # t-test
        M_dir_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Sleep\\ttest_sleep.npy'
        if os.path.exists(M_dir_path) * load:
            M_ttest_sleep = np.load(M_dir_path)
        else:
            M_ttest_sleep = np.zeros((len(self.labels_all), len(self.labels_all), 2, 2)) - 1
            for sc in range(len(self.labels_all)):
                for rc in range(len(self.labels_all)):
                    dat = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1) & (
                            con_trial.Sig > -1)]
                    if len(dat) > 0:  # only feasible connections
                        dat = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1) & (
                                con_trial.Sig >= 0)]

                        if len(dat) > 0:
                            p_W = len(dat.loc[(dat.Sig == 1) & (dat.SleepState == 'Wake')]) / len(
                                dat.loc[(dat.SleepState == 'Wake')])
                            for ss in range(len(sleepstate_labels)):
                                if sleepstate_labels[ss] in sleepstate_labels_u:
                                    ##binomial test
                                    x = np.array(
                                        [len(dat.loc[(dat.Sig == 1) & (dat.SleepState == sleepstate_labels[ss])]),
                                         len(dat.loc[(dat.Sig == 0) & (dat.SleepState == sleepstate_labels[ss])])])
                                    p_bin = scipy.stats.binom_test(x, p=p_W, alternative='two-sided')
                                    if (p_bin < 0.05) & ((x[0] / np.sum(x)) < p_W):
                                        M_ttest_sleep[sc, rc, ss, 1] = 2
                                    elif (p_bin < 0.05) & ((x[0] / np.sum(x)) > p_W):
                                        M_ttest_sleep[sc, rc, ss, 1] = 3
                                    else:
                                        M_ttest_sleep[sc, rc, ss, 1] = 1
                                    ##tt-test
                                    d_NREM, p_NREM = scipy.stats.ttest_ind(
                                        dat.loc[(dat.Sig == 1) & (dat.SleepState == 'Wake'), 'LL'].values,
                                        dat.loc[
                                            (dat.Sig == 1) & (dat.SleepState == sleepstate_labels[ss]), 'LL'].values)
                                    if p_NREM < 0.05:
                                        if np.sign(d_NREM) == 1:
                                            d = 2  # decrease
                                        else:
                                            d = 3  # increase
                                        M_ttest_sleep[sc, rc, ss, 0] = d
                                    else:
                                        M_ttest_sleep[sc, rc, ss, 0] = 1  # no change
                        else:
                            M_ttest_sleep[sc, rc, :, :] = 0  # no significant connections
            np.save(M_dir_path, M_ttest_sleep)
        for ss in range(2):
            M_resp = np.delete(np.delete(M_ttest_sleep[:, :, ss, 0], self.bad_all, 0), self.bad_all, 1)
            M_resp = M_resp * 1
            M_resp = M_resp[self.ind, :]
            M_resp = M_resp[:, self.ind]
            # (M, labels,areas, label, t, method,save= 1, circ = 1, cmap='hot', group='block')
            self.plot_BM(M_resp, self.labels_sel, self.areas_sel, 'ttest_' + sleepstate_labels[ss], 0, 'change', 1,
                         circ=0,
                         group='Sleep')

        for ss in range(2):
            M_resp = np.delete(np.delete(M_ttest_sleep[:, :, ss, 1], self.bad_all, 0), self.bad_all, 1)
            M_resp = M_resp * 1
            M_resp = M_resp[self.ind, :]
            M_resp = M_resp[:, self.ind]
            # (M, labels,areas, label, t, method,save= 1, circ = 1, cmap='hot', group='block')
            self.plot_BM(M_resp, self.labels_sel, self.areas_sel, 'Prob_bin_' + sleepstate_labels[ss], 0, 'change', 1,
                         circ=0,
                         group='Sleep')

    def save_sleep_nmf(self, con_trial, M_Block, M_t_resp):

        ## nnmf input: vecotize
        M_B_nmf = M_Block.reshape(len(M_Block), -1)
        M_B_nmf = M_B_nmf.T
        M_B_nmf[np.isnan(M_B_nmf)] = 0
        M_B_nmf[M_B_nmf < 0] = 0
        # run NMF with rk = 3
        rk = 3
        W, W0, H = NMFf.get_nnmf_Epi(M_B_nmf, rk, it=2000)

        # store sumamry
        start = 0
        for b in np.unique(con_trial.Block).astype('int'):
            h = np.bincount(con_trial.loc[con_trial.Block == b, 'Hour']).argmax()
            s = np.mean(con_trial.loc[
                            con_trial.Block == b, 'Sleep'])  # np.bincount(con_trial.loc[con_trial.Block==b, 'Sleep']).argmax()
            arr = np.zeros((1, rk + 3))
            arr[0, 0] = b
            arr[0, 1] = h
            arr[0, 2] = s > 0
            for i in range(rk):
                arr[0, 3 + i] = H[i, b]
            nmf_arr = pd.DataFrame(arr, columns=['Block', 'Hour', 'Sleep', 'H1', 'H2', 'H3'])
            if start == 0:
                nmf_summary = nmf_arr
                start = 1
            else:
                nmf_summary = pd.concat([nmf_summary, nmf_arr])
        nmf_summary.to_csv(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\NNMF\\nmf_summary_rk' + str(
                rk) + '.csv', index=False, header=True)

        # check whether cluster is stable, increase or decrease
        H_sleep = np.zeros((rk, 3))
        for h, i in zip(['H1', 'H2', 'H3'], np.arange(rk)):
            # sns.catplot('Sleep', h, data=nmf_summary)
            d_NREM, p_NREM = scipy.stats.ttest_ind(nmf_summary.loc[(nmf_summary.Sleep == 0), h],
                                                   nmf_summary.loc[(nmf_summary.Sleep == 1), h])
            H_sleep[i, 0] = i + 1
            if p_NREM < 0.05:
                H_sleep[i, 1] = -1 * np.sign(d_NREM)
                H_sleep[i, 2] = p_NREM
        H_sleep = pd.DataFrame(H_sleep, columns=['H', 'Sleep_change', 'p'])
        if len(np.where(H_sleep.Sleep_change == 0)[0]) == 0:
            val, n_sc = np.unique(H_sleep.Sleep_change, return_counts=True)
            val = val[n_sc >= 2]
            p = np.max(H_sleep.loc[(H_sleep.Sleep_change == val[0]), 'p'].values)
            H_sleep.loc[(H_sleep.Sleep_change == val[0]) & (H_sleep.p == p), 'Sleep_change'] = 0
        # get connections assign to cluster
        W_z = scipy.stats.zscore(W, 0)
        W_pref_z = np.argmax(W_z, 1)
        W_pref_z[np.max(W_z, 1) < 1] = np.where(H_sleep.Sleep_change == 0)[0][0]
        W_pref_z[np.max(W_z, 1) < 0] = -1
        np.savez(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\NNMF\\nmf_rk' + str(
            rk) + '.npz', W, H, H_sleep,
                 W_pref_z)
        ## plotting
        if len(np.unique(H_sleep.Sleep_change)) == 3:

            sleep_con_labels = ['Decrease', 'Stable', 'Increase']
            sleep_con_val = np.arange(-1, 2)
            sleep_plot_val = np.array([2, 1, 3])  # n

            M_sleep_nmf = np.zeros((M_t_resp.shape[0], M_t_resp.shape[0]))
            for lab, val, plot_val in zip(sleep_con_labels, sleep_con_val, sleep_plot_val):
                h = np.where(H_sleep.values[:, 1] == val)[0]

                M_nmf = np.array((W_pref_z == h)).reshape(len(self.labels_all), len(self.labels_all))
                M_nmf = M_nmf * plot_val
                M_sleep_nmf = M_sleep_nmf + M_nmf
            # M_nmf = np.array(((W[:,0]/W[:,1])>3)).reshape(len(labels_all),len(labels_all))
            # M_nmf = np.array(((W[:,0]/W[:,1])<0.8)).reshape(len(labels_all),len(labels_all))
            M_sleep_nmf[M_t_resp[:, :, 1] == -1] = -1
            M_sleep_nmf[M_t_resp[:, :, 1] == 0] = 0
            M_resp = np.delete(np.delete(M_sleep_nmf, self.bad_all, 0), self.bad_all, 1)
            M_resp = M_resp[self.ind, :]
            M_resp = M_resp[:, self.ind]

            self.plot_BM(M_resp, self.labels_sel, self.areas_sel, 'NMF_sleep', 0, 'change', 1, circ=0, group='Sleep')
            for lab, val in zip(sleep_con_labels, sleep_con_val):
                h = np.where(H_sleep.values[:, 1] == val)[0]

                M_nmf = np.array((W_pref_z == h)).reshape(len(self.labels_all), len(self.labels_all))
                # M_nmf = np.array(((W[:,0]/W[:,1])>3)).reshape(len(labels_all),len(labels_all))
                # M_nmf = np.array(((W[:,0]/W[:,1])<0.8)).reshape(len(labels_all),len(labels_all))
                M_resp = np.delete(np.delete(M_nmf * (M_Block[1] > 0), self.bad_all, 0), self.bad_all, 1)
                M_resp = M_resp * 1
                M_resp = M_resp[self.ind, :]
                M_resp = M_resp[:, self.ind]
                # (M, labels,areas, label, t, method,save= 1, circ = 1, cmap='hot', group='block')
                self.plot_BM(M_resp, self.labels_sel, self.areas_sel, 'NMF_' + lab, 0, 'binary', 1, circ=0,
                             group='Sleep')
                # plot_BM_CR_trial_sig(M_resp, labels_sel,areas_sel, ll, t, 'sym',save= 1, circ = 0, group='General')
        else:
            print(H_sleep)
            # plot_BM_CR_trial_sig(M_resp, labels_sel,areas_sel, ll, t, 'sym',save= 1, circ = 0, group='General')

    def NMF_on_blocks(self, con_trial, M_B_all):
        # input Matrix:
        M_B_nmf = M_B_all.reshape(len(M_B_all), -1)  # vecotrize
        M_B_nmf = M_B_nmf.T
        M_B_nmf[np.isnan(M_B_nmf)] = 0
        M_B_nmf[M_B_nmf < 0] = 0

        # get NMF
        rk = 3
        [W, H] = NMFf.get_nnmf_Epi(M_B_nmf, rk, it=2000)
        # for h in range(rk):
        #    plt.plot(H[h])
        # assign whether sleep has an effect:
        start = 0
        for b in np.unique(con_trial.Block).astype('int'):
            h = np.bincount(con_trial.loc[con_trial.Block == b, 'Hour']).argmax()
            s = np.mean(con_trial.loc[
                            con_trial.Block == b, 'Sleep'])  # np.bincount(con_trial.loc[con_trial.Block==b, 'Sleep']).argmax()
            arr = np.zeros((1, rk + 3))
            arr[0, 0] = b
            arr[0, 1] = h
            arr[0, 2] = s > 0  # if there is sleep
            for i in range(rk):
                arr[0, 3 + i] = H[i, b]
            nmf_arr = pd.DataFrame(arr, columns=['Block', 'Hour', 'Sleep', 'H1', 'H2', 'H3'])
            if start == 0:
                nmf_summary = nmf_arr
                start = 1
            else:
                nmf_summary = pd.concat([nmf_summary, nmf_arr])
        # t-test of H coefficients
        H_sleep = np.zeros((rk, 2))
        for h, i in zip(['H1', 'H2', 'H3'], np.arange(rk)):
            sns.catplot('Sleep', h, data=nmf_summary)
            d_NREM, p_NREM = scipy.stats.ttest_ind(nmf_summary.loc[(nmf_summary.Sleep == 0), h],
                                                   nmf_summary.loc[(nmf_summary.Sleep == 1), h])
            H_sleep[i, 0] = i + 1
            if p_NREM < 0.05:
                H_sleep[i, 1] = -1 * np.sign(d_NREM)
        H_sleep = pd.DataFrame(H_sleep, columns=['H', 'Sleep_change'])
        # preferred W
        W_pref_z = stats.zscore(W, axis=0)
        W_pref_z = np.argmax(W_pref_z, 1)
        np.savez(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\NMF\\nmf_rk3.npy', W, H,
                 H_sleep, W_pref_z)

    def get_sleep_summary(self, con_trial, M_t_resp):
        file_con_sleep = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\con_sleep.csv'
        M_dir_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Sleep\\ttest_sleep.npy'
        M_ttest_sleep = np.load(M_dir_path)

        # only LL if there was a significant connection, remove Sig get Prob instead
        con_trial_sig = con_trial[con_trial.d > -10]
        con_trial_sig.loc[con_trial_sig.Sig < 0, 'Sig'] = np.nan
        con_trial_sig.insert(4, 'LL_sig', np.nan)
        con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL_sig'] = con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL']
        con_trial_sig = con_trial_sig.drop(columns='LL')
        con_trial_sig.insert(4, 'LL', con_trial_sig.LL_sig)
        con_trial_sig.insert(4, 'Prob', con_trial_sig.Sig)
        con_trial_sig = con_trial_sig.drop(columns='LL_sig')

        con_trial_sig = con_trial_sig[np.isin(con_trial_sig.Sig, [0, 1])]
        con_sleep = con_trial_sig.groupby(['Stim', 'Chan', 'SleepState'], as_index=False)[['Prob', 'LL', 'd']].mean()
        con_sleep.insert(5, 'ttest_wake', 0)
        con_sleep.insert(5, 'prob_wake', 0)
        con_sleep.insert(5, 't_resp', 0)
        for sc in np.unique(con_sleep.Stim).astype('int'):
            chans = con_sleep.loc[(con_sleep.Stim == sc), 'Chan'].values.astype('int')
            for rc in chans:
                con_sleep.loc[(con_sleep.Stim == sc) & (con_sleep.Chan == rc), 't_resp'] = M_t_resp[sc, rc, 2]
                ix_wake = \
                    np.where(con_sleep.loc[(con_sleep.Stim == sc) & (con_sleep.Chan == rc), 'SleepState'] == 'Wake')[0]
                for ss, s_ix in zip(['NREM', 'REM'], np.arange(2)):
                    ix_sleep = \
                        np.where(con_sleep.loc[(con_sleep.Stim == sc) & (con_sleep.Chan == rc), 'SleepState'] == ss)[0]
                    if (len(ix_wake) > 0) & (len(ix_sleep) > 0):
                        con_sleep.loc[(con_sleep.SleepState == ss) & (con_sleep.Stim == sc) & (
                                con_sleep.Chan == rc), 'ttest_wake'] = M_ttest_sleep[sc, rc, s_ix, 0]

                        con_sleep.loc[(con_sleep.SleepState == ss) & (con_sleep.Stim == sc) & (
                                con_sleep.Chan == rc), 'prob_wake'] = M_ttest_sleep[sc, rc, s_ix, 1]
                con_sleep.loc[
                    (con_sleep.SleepState == 'Wake') & (con_sleep.Stim == sc) & (con_sleep.Chan == rc), 'prob_wake'] = 1
                con_sleep.loc[(con_sleep.SleepState == 'Wake') & (con_sleep.Stim == sc) & (
                        con_sleep.Chan == rc), 'ttest_wake'] = 1  # no change

        dist_groups = np.array([[0, 15], [15, 30], [30, 120]])
        dist_labels = ['local (<15 mm)', 'short (<30mm)', 'long']
        con_sleep.insert(5, 'Dist', 'long')
        for dl, i in zip(dist_labels, np.arange(len(dist_labels))):
            con_sleep.loc[(con_sleep.d < dist_groups[i, 1]) & (con_sleep.d >= dist_groups[i, 0]), 'Dist'] = dist_labels[
                i]

        con_sleep.to_csv(file_con_sleep,
                         index=False,
                         header=True)

    def BM_plots_General(self, M_t_resp, con_trial, reload=0):
        # labels
        # labels:
        labels_sel = np.delete(self.labels_all, self.bad_all, 0)
        areas_sel = np.delete(self.labels_region_L, self.bad_all, 0)
        # sort
        ind = np.argsort(areas_sel)
        areas_sel = np.delete(self.labels_region, self.bad_all, 0)
        labels_sel = labels_sel[ind]
        areas_sel = areas_sel[ind]

        # summary
        con_trial_sig = con_trial[con_trial.d > -10]
        con_trial_sig.insert(4, 'LL_sig', np.nan)
        con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL_sig'] = con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL']
        summ = con_trial_sig[(con_trial_sig.Sig > -1)]  # only possible connections
        summ = summ.groupby(['Stim', 'Chan'], as_index=False)[['Sig', 'LL_sig', 'd']].mean()

        M_dir_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\M_dir.csv'

        if os.path.exists(M_dir_path) * reload:
            print('loading Directionality matrix')
            M_DI = pd.read_csv(M_dir_path)
        else:
            print('calculating Directionality matrix')
            M_DI = np.zeros((1, 5))
            for sc in range(len(self.labels_all)):
                for rc in range(sc, len(self.labels_all)):
                    probA = summ.loc[(summ.Chan == rc) & (summ.Stim == sc), 'Sig'].values
                    if len(probA) == 0:
                        probA = -1
                    else:
                        probA = probA[0]
                    probB = summ.loc[(summ.Chan == sc) & (summ.Stim == rc), 'Sig'].values
                    if len(probB) == 0:
                        probB = -1
                    else:
                        probB = probB[0]
                    M_DI_ix = [[sc, rc, probA, probB, np.nan]]
                    M_DI = np.concatenate([M_DI, M_DI_ix], 0)
            M_DI = M_DI[1:, :]
            M_DI = pd.DataFrame(M_DI, columns=['A', 'B', 'P_AB', 'P_BA', 'DI'])

            M_DI.loc[(M_DI.P_AB > -1) & (M_DI.P_BA > -1), 'DI'] = (1 - np.min(
                [M_DI.loc[(M_DI.P_AB > -1) & (M_DI.P_BA > -1), 'P_AB'],
                 M_DI.loc[(M_DI.P_AB > -1) & (M_DI.P_BA > -1), 'P_BA']], 0) / np.max(
                [M_DI.loc[(M_DI.P_AB > -1) & (M_DI.P_BA > -1), 'P_AB'],
                 M_DI.loc[(M_DI.P_AB > -1) & (M_DI.P_BA > -1), 'P_BA']], 0)) * (M_DI.loc[(M_DI.P_AB > -1) & (
                    M_DI.P_BA > -1), 'P_AB'] - M_DI.loc[(M_DI.P_AB > -1) & (M_DI.P_BA > -1), 'P_BA'])
            M_DI.loc[(M_DI.P_AB > 0) & (M_DI.P_BA == 0), 'DI'] = 1
            M_DI.loc[(M_DI.P_AB == 0) & (M_DI.P_BA > 0), 'DI'] = -1
            M_DI.to_csv(M_dir_path,
                        index=False,
                        header=True)
        asym = np.zeros((len(self.labels_all), len(self.labels_all), 2)) - 1
        for sc in np.unique(M_DI.A).astype('int'):
            dat = M_DI.loc[M_DI.A == sc]
            chans = dat.B.values.astype('int')
            asym[sc, chans, 0] = dat.P_AB.values
            asym[sc, chans, 1] = dat.DI.values
            asym[sc, chans, 0] = asym[sc, chans, 0] + 1 * (abs(dat.DI.values) < 1)
            dat = M_DI.loc[M_DI.B == sc]
            chans = dat.A.values.astype('int')
            asym[sc, chans, 0] = dat.P_BA.values
            asym[sc, chans, 1] = -dat.DI.values
            asym[sc, chans, 0] = asym[sc, chans, 0] + 1 * (abs(dat.DI.values) < 1)
        sc0 = M_DI.loc[(M_DI.P_AB == 0) & (M_DI.P_BA == 0), 'A'].values.astype('int')
        rc0 = M_DI.loc[(M_DI.P_AB == 0) & (M_DI.P_BA == 0), 'B'].values.astype('int')
        for sc, rc in zip(sc0, rc0):
            asym[sc, rc, 1] = -10
            asym[rc, sc, 1] = -10
        asym[asym[:, :, 0] > 1, 0] = 2  # set probabilities to 1 if higher tha 0
        asym[(asym[:, :, 0] < 2) & (asym[:, :, 0] > 0), 0] = 1  # set probabilities to 1 if higher tha 0
        # asym[(asym[:, :, 0] ==0), 1] = -10
        asym[np.isnan(asym[:, :, 1]), 1] = -20  # set probabilities to 1 if higher tha 0

        ##save binary and symmetry plots
        M_resp = np.delete(np.delete(asym[:, :, 0], self.bad_all, 0), self.bad_all, 1)
        M_resp = M_resp[ind, :]
        M_resp = M_resp[:, ind]
        ll = 'General'  # +', '+sleep_states[s]
        self.plot_BM(M_resp, labels_sel, areas_sel, ll, 0, 'sym', save=1, circ=0, group='General')
        M_resp_b = np.copy(M_resp)
        M_resp_b[M_resp_b > 0] = 1
        self.plot_BM(M_resp_b, labels_sel, areas_sel, ll, 0, 'binary', save=1, circ=0, group='General')

        ##save binary and symmetry plots
        M_resp = np.delete(np.delete(asym[:, :, 1], self.bad_all, 0), self.bad_all, 1)
        M_resp = M_resp[ind, :]
        M_resp = M_resp[:, ind]
        ll = 'General'  # +', '+sleep_states[s]
        self.plot_BM(M_resp, labels_sel, areas_sel, ll, 0, 'Dir', save=1, circ=0, group='General')
        asym[asym < -9] = np.nan

        ##

        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\summary_general.csv'
        if os.path.exists(summary_gen_path) * reload:
            # print('loading')
            summ = pd.read_csv(summary_gen_path)
        else:
            # print('calculating')
            # aymmetry and probability by distance

            summ.insert(4, 't_resp', -1)
            summ.insert(4, 'Dir_B', -1)
            summ.insert(4, 'Dir_index', -1)
            # asym[rc, sc, 1]

            for sc in np.unique(summ.Stim).astype('int'):
                for rc in np.unique(summ.Chan).astype('int'):
                    summ.loc[(summ.Stim == sc) & (summ.Chan == rc), 't_resp'] = M_t_resp[sc, rc, 2]
                    summ.loc[(summ.Stim == sc) & (summ.Chan == rc), 'Dir_B'] = asym[sc, rc, 0]
                    summ.loc[(summ.Stim == sc) & (summ.Chan == rc), 'Dir_index'] = asym[sc, rc, 1]
                    if asym[sc, rc, 1] == -10:
                        print(sc, rc)
            dist_groups = np.array([[0, 15], [15, 30], [30, 120]])
            dist_labels = ['local (<15 mm)', 'short (<30mm)', 'long']
            for dl, i in zip(dist_labels, np.arange(len(dist_labels))):
                summ.loc[(summ.d < dist_groups[i, 1]) & (summ.d >= dist_groups[i, 0]), 'Dist'] = dist_labels[i]
            summ.to_csv(summary_gen_path, index=False, header=True)
        # plots
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('xkcd:white')
        sns.swarmplot(x='Dist', y='Sig', hue='Dir_B', data=summ[summ.Dir_index > 0], palette=["#c34f2f", "#1e4e79"])

        plt.ylim([0, 1])
        plt.legend(['Bi-directional', 'Uni-drectional'], fontsize=15)
        plt.ylabel('Probability', fontsize=25)
        plt.title('Across all trials', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('')
        plt.savefig(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\General\\Sym_dist_scatter.svg')
        plt.show()
        # sns.scatterplot(x='d', y='t_resp', hue='Dir', data=summ[summ.Dir>0])
        # plt.legend(['Uni-drectional', 'Bi-directional'])
        # plt.show()
        # sns.catplot(x='Dist', y= 't_resp', hue='Dir', data=summ[summ.Dir>0], kind='swarm', aspect=3)
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('xkcd:white')
        sns.histplot(x='Dist', hue='Dir_B', data=summ[summ.Dir_index > 0], multiple="stack",
                     palette=["#c34f2f", "#1e4e79"])
        plt.legend(['Bi-directional', 'Uni-drectional'], fontsize=15)
        plt.ylabel('Number of significant connections', fontsize=25)
        plt.title('Across all trials', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('')
        plt.savefig(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\General\\Sym_dist_bar.svg')
        plt.show()
        ## prob and LL
        M_LL = np.zeros((len(self.labels_all), len(self.labels_all), 2)) - 1
        for sc in np.unique(summ.Stim).astype('int'):
            dat = summ.loc[(summ.Stim == sc)]
            chans = dat.Chan.values.astype('int')

            prob = dat.Sig.values
            prob[prob < 0] = -1
            LL = dat.LL_sig.values
            # prob[np.isnan(LL)] = -1

            LL[np.isnan(LL)] = 0
            M_LL[sc, chans, 0] = LL
            M_LL[sc, chans, 1] = prob
        for method, i in zip(['LL', 'Prob'], np.arange(2)):
            asym = M_LL[:, :, i]
            M_resp = np.delete(np.delete(asym, self.bad_all, 0), self.bad_all, 1)
            M_resp = M_resp[ind, :]
            M_resp = M_resp[:, ind]
            ll = 'General'  # +', '+sleep_states[s]
            self.plot_BM(M_resp, labels_sel, areas_sel, ll, 0, method, save=1, circ=0, group='General')


def start_subj(subj, sig=0):
    print(subj + ' -- START --')
    run_main = main(subj)
    path_patient_analysis = 'y:\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj

    # file_t_resp = path_patient_analysis + '\\' + folder + '\\data\\M_t_resp.npy'
    file_t_resp = path_patient_analysis + '\\' + folder + '\\data\\M_tresp.npy'  # for each connection: LLsig (old), t_onset (old), t_resp, CC_p, CC_LL1, CC_LL2
    file_GT = path_patient_analysis + '\\' + folder + '\\data\\M_CC.npy'
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    file_sig_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\sig_con.csv'
    file_CC_surr = path_patient_analysis + '\\' + folder + '\\data\\M_CC_surr.csv'

    # todo: make clean
    con_trial = pd.read_csv(file_con)
    M_GT_all = np.load(file_GT)
    M_t_resp = np.load(file_t_resp)
    sig_con = pd.read_csv(file_sig_con)
    surr_thr = pd.read_csv(file_CC_surr)
    if "Sig_CC_LL" not in sig_con:
        sig_con.insert(3, 'Sig_CC_LL', 0)
    sig_con.Sig_CC_LL = 0
    sig_con.loc[sig_con.Sig_LL == -1, 'Sig_CC_LL'] = -1
    for rc in range(M_GT_all.shape[0]):
        thr = 1.1 * surr_thr.loc[surr_thr.Chan == rc, 'CC_LL99'].values[0]
        # thr = np.min([0.65, thr])
        sig_con.loc[(sig_con.Chan == rc) & ((sig_con.CC_LL1 >= thr) | (sig_con.CC_LL2 >= thr)), 'Sig_CC_LL'] = 1

    if "LL_onset" not in con_trial:
        con_trial.insert(4, 'LL_onset', 0)

    if 'd' not in sig_con:
        # sig_con.insert(3, 'd', 0)
        summ_d = con_trial.groupby(['Stim', 'Chan'], as_index=False)[['d']].mean()
        sig_con = pd.merge(sig_con, summ_d, on=['Stim', 'Chan'])
        sig_con.to_csv(file_sig_con,
                       index=False,
                       header=True)

    if "Sig" not in con_trial:
        sig = 1
    if sig:
        for col in ['t_N2', 'N2', 'sN1', 'sN2', 'N1', 't_N1', 'Sig', 'PLL', 'Pearson', 'Sig']:
            if col in con_trial: con_trial = con_trial.drop(columns=col)
        con_trial.insert(5, 'Sig', -1)
        EEG_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy'
        EEG_CR = np.load(EEG_CR_file)
        remove_art = 1
        if remove_art:
            for rc in range(len(EEG_CR)):
                tt = np.where(np.max(abs(EEG_CR[rc, :, :500]), 1) > 3000)[0]
                con_trial.loc[(con_trial.Chan == rc) & np.isin(con_trial.Num, tt), 'Artefact'] = 2
            con_trial.to_csv(file_con,
                             index=False,
                             header=True)

        for sc in tqdm.tqdm(np.unique(con_trial.Stim), desc='Stimulation Channel'):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Artefact == 0) & (con_trial.Stim == sc), 'Chan']).astype(
                'int')
            for rc in resp_chans:
                dat = sig_con.loc[(sig_con.Stim == sc) & (sig_con.Chan == rc)]
                if dat.Sig_CC_LL.values[0]:
                    if subj == 'EL009':
                        thr = np.min([2, 1.1 * surr_thr.loc[surr_thr.Chan == rc, 'CC_LL99'].values[0]])
                    else:
                        thr = 1.1 * surr_thr.loc[surr_thr.Chan == rc, 'CC_LL99'].values[0]
                    M_GT = M_GT_all[sc, rc, :, :]
                    if len(np.where(dat[['CC_LL1', 'CC_LL2']].values[0][:] < thr)[0][:] + 1) > 0:
                        M_GT = np.delete(M_GT, np.where(dat[['CC_LL1', 'CC_LL2']].values[0][:] < thr)[0][:] + 1, 0)
                    # M_GT[np.where(dat[['CC_LL1','CC_LL2']].values[0][:]<thr)[0][:]+1,:] = np.nan
                    t_resp = dat.t_onset.values[0]
                    con_trial = run_main.get_sig(sc, rc, con_trial, M_GT, t_resp, EEG_CR, p=90, exp=2,
                                                 w_cluster=0.25)
                else:
                    con_trial.loc[(con_trial.Chan == rc) & (con_trial.Stim == sc), 'Sig'] = 0
        con_trial.to_csv(file_con,
                         index=False,
                         header=True)

    if not 'SleepState' in con_trial:
        con_trial.insert(5, 'SleepState', 'Wake')
    con_trial.loc[(con_trial.Sleep == 0), 'SleepState'] = 'Wake'
    con_trial.loc[(con_trial.Sleep > 0) & (con_trial.Sleep < 4), 'SleepState'] = 'NREM'
    con_trial.loc[(con_trial.Sleep == 4), 'SleepState'] = 'REM'
    con_trial.to_csv(file_con,
                     index=False,
                     header=True)

    general = 1
    if general:
        run_main.BM_plots_General(M_t_resp, con_trial)
    blocks = 1
    if blocks:
        # Blockwise BM
        _ = run_main.save_M_block(con_trial, metrics=['LL'], savefig=1)
        # np.save(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\M_B_all.npy', M_B_all)
    sleep = 0
    sleep_nmf = 0
    if (subj == 'EL013') | (subj == 'EL012'):  # not enough sleep data
        sleep = 0
        sleep_nmf = 0

    if sleep:
        run_main.save_M_sleep(con_trial, metrics=['LL', 'Prob'], savefig=1)
        # sleep ttest
        run_main.get_sleep_ttest(con_trial, load=0)
        run_main.get_sleep_summary(con_trial, M_t_resp)

    if sleep_nmf:
        M_dir_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\BM_figures\\Block\\BM_LL.npy'
        M_Block = np.load(M_dir_path)
        run_main.save_sleep_nmf(con_trial, M_Block, M_t_resp)

    print(subj + ' ----- DONE')


thread = 0
sig = 0
# todo: 'EL009',
for subj in ['EL018']:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'
    if thread:
        _thread.start_new_thread(start_subj, (subj, sig))
    else:
        start_subj(subj, 0)
if thread:
    while 1:
        time.sleep(1)
