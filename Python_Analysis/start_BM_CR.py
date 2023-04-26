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
import BM_stats
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
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


class main:
    def __init__(self, subj):
        #  basics, get 4s of data for each stimulation, [-2,2]s
        self.folder = 'BrainMapping'
        self.cond_folder = 'CR'
        self.path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
        path_gen = os.path.join(sub_path + '\\Patients\\' + subj)
        if not os.path.exists(path_gen):
            path_gen = 'T:\\EL_experiment\\Patients\\' + subj
        path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
        path_infos = os.path.join(path_gen, 'Electrodes')
        if not os.path.exists(os.path.join(path_infos, subj + "_labels.xlsx")):
            path_infos = os.path.join(path_gen, 'infos')
        if not os.path.exists(path_infos):
            path_infos = path_gen + '\\infos'

        self.Fs = 500
        self.dur = np.zeros((1, 2), dtype=np.int32)
        self.dur[0, :] = [-1, 3]
        self.dur_tot = np.int32(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        # load patient specific information
        lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
        if "type" in lbls:
            lbls = lbls[lbls.type == 'SEEG']
            lbls = lbls.reset_index(drop=True)
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
        atlas_regions = pd.read_excel(
            "X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients\\Across\\elab_labels.xlsx",
            sheet_name="atlas")
        self.labels_region = labels_region
        for i in range(len(labels_all)):
            area_sel = " ".join(re.findall("[a-zA-Z_]+", labels_all[i]))
            self.labels_region[i] = atlas_regions.loc[atlas_regions.Abbreviation == area_sel, "Region"].values[0]
        # self.labels_region = labels_region

        # regions information
        self.CR_color = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\Analysis\BrainMapping\CR_color.xlsx",
                                      header=0)
        regions = pd.read_excel(sub_path + "\\EvM\Projects\EL_experiment\Analysis\Patients\Across\elab_labels.xlsx",
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
        StimChanIx = np.unique(np.array(StimChanIx).astype('int'))
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

        fig = plt.figure(figsize=(25, 25))
        fig.patch.set_facecolor('xkcd:white')
        axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])  # x, y, (start posiion), lenx, leny
        if method == 'LL':
            cmap = copy.copy(plt.cm.get_cmap(cmap))  # cmap = copy.copy(mpl.cm.get_cmap(cmap))
            cmap.set_under('#2b2b2b')
            cmap.set_bad('black')
            M = np.ma.masked_equal(M, 0)
            im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=6)  # np.percentile(M, 95)
        elif method == 'Prob':
            cmap = copy.copy(plt.cm.get_cmap(cmap))  # cmap = copy.copy(mpl.cm.get_cmap(cmap))
            cmap.set_under('#2b2b2b')
            cmap.set_bad('black')
            M = np.ma.masked_equal(M, 0)
            im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=0.95)
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
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=18)
        plt.yticks(range(len(labels)), labels, fontsize=18)
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

    def save_M_block(self, con_trial, metrics=['LL'], savefig=1):
        con_trial_sig = con_trial[(con_trial.Sleep < 5) & (con_trial.d > -10)]
        con_trial_sig = con_trial_sig.reset_index(drop=True)
        con_trial_sig.loc[con_trial_sig.Sig < 0, 'Sig'] = np.nan
        con_trial_sig.insert(4, 'LL_sig', np.nan)
        con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL_sig'] = con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL']
        con_trial_sig = con_trial_sig.drop(columns='LL')
        con_trial_sig.insert(4, 'LL', con_trial_sig.LL_sig)
        con_trial_sig.insert(4, 'Prob', con_trial_sig.Sig)
        con_trial_sig = con_trial_sig.drop(columns='LL_sig')
        con_trial_sig = con_trial_sig[(con_trial_sig.Block >= 1)]
        con_trial_sig = con_trial_sig.reset_index(drop=True)
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
            M_B_allp = np.zeros((int(np.max(con_trial_sig.Block) + 1), len(self.labels_all), len(self.labels_all)))
            M_B_allp[:, :, :] = M_B_all[:, :, :]
            M_B_allp[np.isnan(M_B_allp)] = 0
            M_B_pear = np.zeros((int(np.max(con_trial_sig.Block) + 1), int(np.max(con_trial_sig.Block) + 1)))
            M_pearson = np.zeros((1, 9))
            block_all = np.unique(con_trial_sig.Block).astype('int')
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
        ## BM plot for LL and Prob for each sleep state
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

        sleepstate_labels = ['Wake', 'NREM', 'REM']  # np.unique(con_trial_sig['SleepState'])

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

    def get_sleep_ttest_surr(self, con_trial_true, load=1):
        con_trial = con_trial_true.copy()
        np.random.shuffle(con_trial['SleepState'].values)
        sleepstate_labels_u = np.unique(con_trial.loc[con_trial.Sleep > 0, 'SleepState'])
        sleepstate_labels = ['NREM', 'REM']
        # t-test
        M_dir_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Sleep\\ttest_sleep_surr.npy'
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

        file_con_sleep = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\con_sleep_surr.csv'

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

        for sc in np.unique(con_sleep.Stim).astype('int'):
            chans = con_sleep.loc[(con_sleep.Stim == sc), 'Chan'].values.astype('int')
            for rc in chans:
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

        con_sleep.to_csv(file_con_sleep,
                         index=False,
                         header=True)

    def get_sleep_surr(self, con_trial, surr=1, again=1):

        # calculate cohens d (LL) and PRob ratio to wake and compares it to surrogates (randomizing labels)
        file_con_sleep = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\con_sleep_stats.csv'

        if again:  # not os.path.exists(file_con_sleep):
            con_trial_sig = con_trial.copy()
            con_trial_sig.loc[con_trial_sig.Sig < 0, 'Sig'] = np.nan
            con_trial_sig.insert(4, 'LL_sig', np.nan)
            con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL_sig'] = con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL']
            con_trial_sig = con_trial_sig.drop(columns='LL')
            con_trial_sig.insert(4, 'LL', con_trial_sig.LL_sig)
            con_trial_sig.insert(4, 'Prob', con_trial_sig.Sig)
            con_trial_sig = con_trial_sig.drop(columns=['LL_sig'])
            con_trial_sig = con_trial_sig[np.isin(con_trial_sig.Sig, [0, 1])]
            con_trial_sig.insert(5, 'SS', con_trial_sig.SleepState)
            con_sleep = con_trial_sig.groupby(['Stim', 'Chan', 'SleepState'], as_index=False)[
                ['Prob', 'LL', 'd']].mean()
            con_sleep = con_sleep.reset_index(drop=True)
            con_sleep.insert(5, 'P_ratio_sig', np.nan)
            con_sleep.insert(5, 'LL_cd_sig', np.nan)
            con_sleep.insert(5, 'P_ratio', np.nan)
            con_sleep.insert(5, 'LL_cd', np.nan)

            all_con = con_sleep.groupby(['Stim', 'Chan'], as_index=False)[['Prob']].mean()
            all_con = all_con[all_con.Prob > 0]
            print('Calculate Ratio and LL difference and surrogates for each connection: ')
            for sc, rc in tqdm.tqdm(zip(all_con.Stim.values.astype('int'), all_con.Chan.values.astype('int')),
                                    total=len(all_con.Stim.values.astype('int'))):
                dat = con_trial_sig[
                    (con_trial_sig.Stim == sc) & (con_trial_sig.Chan == rc) & (con_trial_sig.Artefact < 1)]
                dat = dat.reset_index(drop=True)
                p_W = np.mean(dat.loc[dat.SleepState == 'Wake', 'Prob'])
                LL_w = dat.loc[(dat.Sig == 1) & (dat.SleepState == 'Wake'), 'LL'].values
                ix_wake = \
                    np.where(con_sleep.loc[(con_sleep.Stim == sc) & (con_sleep.Chan == rc), 'SleepState'] == 'Wake')[0]
                for ss, s_ix in zip(['NREM', 'REM'], np.arange(2)):
                    ix_sleep = \
                        np.where(con_sleep.loc[(con_sleep.Stim == sc) & (con_sleep.Chan == rc), 'SleepState'] == ss)[0]
                    if (len(ix_wake) > 0) & (len(ix_sleep) > 0):
                        LL_s = dat.loc[
                            (dat.Sig == 1) & (dat.SleepState == ss), 'LL'].values
                        p_SS = np.mean(dat.loc[dat.SleepState == ss, 'Prob'])
                        # prob ratio
                        r = np.sign(p_SS - p_W) * (1 - np.min([p_SS, p_W]) / np.max([p_SS, p_W]))
                        # sig_thr = BM_stats.R_surr(dat, feature_states=['Wake', ss])
                        con_sleep.loc[(con_sleep.SleepState == ss) & (con_sleep.Stim == sc) & (
                                con_sleep.Chan == rc), 'P_ratio'] = r
                        # cohen's d
                        cd = BM_stats.cohen_d(LL_s, LL_w)
                        con_sleep.loc[(con_sleep.SleepState == ss) & (con_sleep.Stim == sc) & (
                                con_sleep.Chan == rc), 'LL_cd'] = cd
                        # surr
                        if surr:
                            n = 100
                            p = 5
                            surr_cd = np.zeros((n, 2))
                            for i in range(n):
                                np.random.shuffle(dat['SS'].values)
                                p_SS_s = np.mean(dat.loc[(dat['SS'] == ss), 'Prob'])
                                p_W_s = np.mean(
                                    dat.loc[(dat['SS'] == 'Wake'), 'Prob'])

                                surr_cd[i, 0] = np.sign(p_SS_s - p_W_s) * (
                                        1 - np.min([p_SS_s, p_W_s]) / np.max([p_SS_s, p_W_s]))
                                surr_cd[i, 1] = BM_stats.cohen_d(dat.loc[
                                                                     (dat.Sig == 1) & (dat['SS'] == ss), 'LL'].values,
                                                                 dat.loc[
                                                                     (dat.Sig == 1) & (
                                                                             dat['SS'] == 'Wake'), 'LL'].values)

                            # sig_thr = BM_stats.CD_surr(dat[dat.Sig == 1], feature_states=['Wake', ss])
                            con_sleep.loc[(con_sleep.SleepState == ss) & (con_sleep.Stim == sc) & (
                                    con_sleep.Chan == rc), 'LL_cd_sig'] = (cd < np.percentile(surr_cd[:, 1], p)) | (
                                    cd > np.percentile(surr_cd[:, 1], 100 - p))
                            con_sleep.loc[(con_sleep.SleepState == ss) & (con_sleep.Stim == sc) & (
                                    con_sleep.Chan == rc), 'P_ratio_sig'] = (r < np.percentile(surr_cd[:, 0], p)) | (
                                    r > np.percentile(surr_cd[:, 0], 100 - p))

                con_sleep.loc[
                    (con_sleep.SleepState == 'Wake') & (con_sleep.Stim == sc) & (con_sleep.Chan == rc), 'P_ratio'] = 1
                con_sleep.loc[(con_sleep.SleepState == 'Wake') & (con_sleep.Stim == sc) & (
                        con_sleep.Chan == rc), 'LL_cd'] = 0  # no change

            dist_groups = np.array([[0, 15], [15, 30], [30, 120]])
            dist_labels = ['local (<15 mm)', 'short (<30mm)', 'long']
            con_sleep.insert(5, 'Dist', 'long')
            for dl, i in zip(dist_labels, np.arange(len(dist_labels))):
                con_sleep.loc[(con_sleep.d < dist_groups[i, 1]) & (con_sleep.d >= dist_groups[i, 0]), 'Dist'] = \
                    dist_labels[
                        i]

            con_sleep.to_csv(file_con_sleep,
                             index=False,
                             header=True)

    def get_sleep_ttest(self, con_trial, load=1):
        sleepstate_labels_u = np.unique(con_trial.loc[con_trial.Sleep > 0, 'SleepState'])
        sleepstate_labels = ['NREM', 'REM']
        # t-test
        M_dir_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Sleep\\ttest_sleep.npy'
        if os.path.exists(M_dir_path) * load:
            M_ttest_sleep = np.load(M_dir_path)
        else:
            M_ttest_sleep = np.zeros((len(self.labels_all), len(self.labels_all), len(sleepstate_labels), 2)) - 1
            for sc in range(len(self.labels_all)):
                for rc in range(len(self.labels_all)):
                    dat = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1) & (
                            con_trial.Sig > -1)]
                    dat = dat.reset_index(drop=True)

                    if len(dat) > 0:  # only feasible connections
                        dat = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1) & (
                                con_trial.Sig >= 0)]
                        dat = dat.reset_index(drop=True)

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
        con_trial = con_trial[con_trial.Block >= 1]
        con_trial = con_trial.reset_index(drop=True)
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
                # con_sleep.loc[(con_sleep.Stim == sc) & (con_sleep.Chan == rc), 't_resp'] = M_t_resp[sc, rc, 2]
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

        con_sleep.to_csv(file_con_sleep,
                         index=False,
                         header=True)

    def BM_plots_General(self, CC_summ, con_trial, reload=0):
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
        con_trial_sig = con_trial_sig.reset_index(drop=True)
        con_trial_sig.insert(4, 'LL_sig', np.nan)

        con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL_sig'] = con_trial_sig.loc[con_trial_sig.Sig == 1, 'LL']

        summ = con_trial_sig[(con_trial_sig.Sig > -1)]  # only possible connections
        summ = summ.reset_index(drop=True)
        summ = summ.groupby(['Stim', 'Chan'], as_index=False)[['Sig', 'LL_sig', 'd']].mean()
        M_dir_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\M_DI.csv'  # c M_dir

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

        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\summ_general.csv'  # summary_general
        if os.path.exists(summary_gen_path) * 0:
            # print('loading')
            summ = pd.read_csv(summary_gen_path)
        else:
            CC_summ = CC_summ[(CC_summ.art == 0) & (CC_summ.sig_w == 1)]
            CC_summ = CC_summ.groupby(['Stim', 'Chan'], as_index=False)[['t_WOI', 'onset']].mean()
            summ = pd.merge(summ, CC_summ, on=['Stim', 'Chan'], how='outer')
            summ.insert(4, 'DI', -1)  # asym[rc, sc, 1]
            for sc in np.unique(summ.Stim).astype('int'):
                for rc in np.unique(summ.Chan).astype('int'):
                    summ.loc[(summ.Stim == sc) & (summ.Chan == rc), 'DI'] = asym[sc, rc, 1]
            dist_groups = np.array([[0, 15], [15, 30], [30, 500]])
            dist_labels = ['local (<15 mm)', 'short (<30mm)', 'long']
            for dl, i in zip(dist_labels, np.arange(len(dist_labels))):
                summ.loc[(summ.d < dist_groups[i, 1]) & (summ.d >= dist_groups[i, 0]), 'Dist'] = dist_labels[i]
            summ.to_csv(summary_gen_path, index=False, header=True)
        # plots

        summ.insert(4, 'Dir_B', 0)
        summ.loc[summ.DI > 0.5, 'Dir_B'] = 1
        summ.loc[summ.DI > 0.5, 'Dir_B'] = 1
        summ.loc[np.isnan(summ.DI), 'Dir_B'] = 2
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('xkcd:white')
        sns.swarmplot(x='Dist', y='Sig', hue='Dir_B', data=summ[summ.Sig > 0])

        plt.ylim([0, 1.1])
        plt.legend(['Bi-directional', 'Uni-drectional', 'Unknown'], fontsize=15)
        plt.ylabel('Probability', fontsize=25)
        plt.title('Across all trials', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('')
        plt.savefig(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\General\\Sym_dist_scatter.svg')
        plt.show()

        ## prob and LL
        M_LL = np.zeros((len(self.labels_all), len(self.labels_all), 2)) - 1
        for sc in np.unique(summ.Stim).astype('int'):
            dat = summ.loc[(summ.Stim == sc)]
            chans = dat.Chan.values.astype('int')

            prob = dat.Sig.values
            prob[prob < 0] = -1
            LL = dat.LL_sig.values
            prob[np.isnan(LL)] = 0

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


def start_subj(subj, cluster_method='similarity', sig=0):
    print(subj + ' -- START --')
    run_main = main(subj)
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    # load data
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    # file_CC_summ = path_patient_analysis + '\\' + folder + '\\data\\CC_summ.csv'
    file_CC_summ = path_patient_analysis + '\\' + folder + '\\data\\CC_summ_' + cluster_method + '.csv'
    # todo: make clean
    con_trial = pd.read_csv(file_con)
    CC_summ = pd.read_csv(file_CC_summ)

    if not 'SleepState' in con_trial:
        con_trial.insert(6, 'SleepState', 'Wake')
    con_trial.loc[(con_trial.SleepState == 'W'), 'SleepState'] = 'Wake'
    con_trial.loc[(con_trial.Sleep == 0), 'SleepState'] = 'Wake'
    con_trial.loc[(con_trial.Sleep > 1) & (con_trial.Sleep < 4), 'SleepState'] = 'NREM'
    con_trial.loc[(con_trial.Sleep == 1), 'SleepState'] = 'NREM1'
    con_trial.loc[(con_trial.Sleep == 6), 'SleepState'] = 'SZ'
    con_trial.loc[(con_trial.Sleep == 4), 'SleepState'] = 'REM'


    general = 1
    if general:
        run_main.BM_plots_General(CC_summ, con_trial, 0)
    blocks = 1
    if blocks:
        # Blockwise BM
        _ = run_main.save_M_block(con_trial, metrics=['LL'], savefig=1)
        # np.save(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\M_B_all.npy', M_B_all)
    sleep = 1
    sleep_nmf = 1
    if (subj == 'EL013') | (subj == 'EL012') | (subj == 'EL021')| (subj == 'EL022'):  # not enough sleep data
        sleep = 0
        sleep_nmf = 0

    if sleep:
        run_main.save_M_sleep(con_trial, metrics=['LL', 'Prob'], savefig=1)
        # sleep ttest
        run_main.get_sleep_surr(con_trial, surr=1)
        # run_main.get_sleep_ttest(con_trial, load=0)
        # run_main.get_sleep_summary(con_trial, M_t_resp)

    # if sleep_nmf:
    #     M_dir_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\BM_figures\\Block\\BM_LL.npy'
    #     M_Block = np.load(M_dir_path)
    #     run_main.save_sleep_nmf(con_trial, M_Block, M_t_resp)

    print(subj + ' ----- DONE')


# thread = 0
# sig = 0
# # todo: 'EL009',
# for subj in [ 'EL013', 'EL014', "EL015", "EL016", "EL017", "EL019","EL020", "EL021"]:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'"EL021", "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016",
#     if thread:
#         _thread.start_new_thread(start_subj, (subj, sig))
#     else:
#         start_subj(subj,'similarity', 0)
# if thread:
#     while 1:
#         time.sleep(1)
