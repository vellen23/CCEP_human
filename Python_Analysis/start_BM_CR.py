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
from scipy import stats
from matplotlib.patches import Rectangle
import freq_funcs as ff
# from tqdm.notebook import trange, tqdm
# remove some warnings
import warnings
from pathlib import Path
import LL_funcs as LLf
import copy
import h5py
import BM_func as BMf
import graphNMF
import BM_plots
import graph_funcs
import load_summary as ls

# I expect to see RuntimeWarnings in this block
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sleepstate_labels = ['NREM', 'REM', 'Wake']

folder = 'BrainMapping'
cond_folder = 'CR'
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])


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
        self.hemisphere = lbls.Hemisphere
        stimlist = pd.read_csv(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\stimlist_' + self.cond_folder + '.csv')
        if len(stimlist) == 0:
            stimlist = pd.read_csv(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\stimlist_' + self.cond_folder + '.csv')
        #
        labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
            stimlist,
            lbls)
        bad_region = np.where((labels_region == 'Unknown') | (labels_region == 'WM') | (labels_region == 'OUT') | (
                labels_region == 'Putamen'))[0]
        self.labels_region_L = lbls.Hemisphere.values + '_' + labels_region
        self.subj = subj
        self.lbls = lbls
        atlas_regions = pd.read_excel(
            "X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients\\Across\\elab_labels.xlsx",
            sheet_name="atlas")
        self.labels_region = labels_region
        for i in range(len(labels_all)):
            area_sel = " ".join(re.findall("[a-zA-Z_]+", labels_all[i]))
            self.labels_region[i] = atlas_regions.loc[atlas_regions.Abbreviation == area_sel, "Region"].values[0]
        # self.labels_region = labels_region

        # regions information
        # self.CR_color = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\Analysis\BrainMapping\CR_color.xlsx",
        #                               header=0)
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

    def get_summary(self, con_trial, CC_summ, EEG_resp):
        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\summ_general.csv'  # summary_general
        con_summary = BMf.get_con_summary(con_trial, CC_summ, EEG_resp)
        con_summary.to_csv(summary_gen_path, index=False, header=True)  # get_con_summary_wake

    def get_summary_SS(self, con_trial, CC_summ):
        con_trial = bf.add_sleepstate(con_trial)
        for ss in ['Wake', 'NREM', 'REM']:
            summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\summ_' + ss + '.csv'  # summary_general
            df = BMf.get_con_summary_SS(con_trial, CC_summ, ss)
            df = ls.adding_area(df, self.lbls, pair=1)
            df = ls.adding_region(df, pair=1)
            df = ls.adding_subregion(df, pair=1)
            df.to_csv(summary_gen_path, index=False, header=True)  # get_con_summary_wake

    def get_node_features(self, con_trial, skip=1):
        exp_dir = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'Graph', 'Node')
        os.makedirs(exp_dir, exist_ok=True)
        file = os.path.join(exp_dir, 'node_features_sleep.csv')
        if os.path.isfile(file) * skip:
            print('Node features already calculated  -  skipping .. ')
            df = pd.read_csv(file)
            df.Chan = df.Chan.astype('int')
        else:
            df = graph_funcs.node_features_sleep_trial(con_trial, file)
            df.insert(0, 'Subj', self.subj)
        df = ls.adding_area(df, self.lbls, pair=0)
        df = ls.adding_subregion(df, pair=0)
        df = ls.adding_region(df, pair=0)
        df.to_csv(file, header=True, index=False)
        print('Node features calculated')

    def connection_sleep_diff(self, con_trial, sig=1, skip=1):
        exp_dir = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'Graph', 'Connection')
        os.makedirs(exp_dir, exist_ok=True)
        if sig:
            file = os.path.join(exp_dir, 'con_sleep_stats.csv')
        else:
            file = os.path.join(exp_dir, 'con_sleep_stats_LL.csv')
        if os.path.isfile(file) * skip:
            df = pd.read_csv(file)
            df.Chan = df.Chan.astype('int')
        else:
            df = graph_funcs.con_sleep_stats(con_trial, sig)
            df.insert(0, 'Subj', self.subj)
        df = ls.adding_area(df, self.lbls, pair=1)
        df = ls.adding_region(df, pair=1)
        df = ls.adding_subregion(df, pair=1)
        df.to_csv(file, header=True, index=False)
        print('Node features calculated')

    def plot_BM_sleep(self, SleepState='NREM'):
        exp_dir = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'Graph', 'Connection')
        os.makedirs(exp_dir, exist_ok=True)
        file = os.path.join(exp_dir, 'con_sleep_stats.csv')
        df = pd.read_csv(file)
        BM_W = np.zeros((len(self.labels_all), len(self.labels_all))) - 2
        # connection with significan tnegative effect
        chans = df.loc[(df.SleepState == SleepState) & (df.Sig == 1) & (df.biserial < 0), 'Chan'].values.astype('int')
        stim = df.loc[(df.SleepState == SleepState) & (df.Sig == 1) & (df.biserial < 0), 'Stim'].values.astype('int')
        BM_W[stim, chans] = -1
        # connection with significant positive effect
        chans = df.loc[(df.SleepState == SleepState) & (df.Sig == 1) & (df.biserial > 0), 'Chan'].values.astype('int')
        stim = df.loc[(df.SleepState == SleepState) & (df.Sig == 1) & (df.biserial > 0), 'Stim'].values.astype('int')
        BM_W[stim, chans] = 1
        # connection with no sig.effect
        chans = df.loc[(df.SleepState == SleepState) & (df.Sig == 0), 'Chan'].values.astype('int')
        stim = df.loc[(df.SleepState == SleepState) & (df.Sig == 0), 'Stim'].values.astype('int')
        BM_W[stim, chans] = 0

        BM_W = BM_W.astype('int')
        cmap = plt.get_cmap('seismic', 3)  # matplotlib.colormaps["Accent"]
        cmap.set_bad(color='black')
        # mask some 'bad' data, in your case you would have: data == 0
        BM_W = np.ma.masked_where(BM_W == -2, BM_W)

        fig = plt.figure(figsize=(10, 10))
        plt.suptitle('Clusters - Sleep')
        axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        axcolor = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        BM_plots.plot_BM(BM_W, self.labels_all, self.hemisphere, axmatrix, axcolor=axcolor, cmap=cmap,
                         vlim=[np.min(BM_W) - 0.5, np.max(BM_W) + 0.5], sort=1, cat=1)
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'BM_figures', 'Sleep')
        plt.savefig(os.path.join(path_output, 'BM_' + SleepState + '_Effect.svg'))
        plt.show()

    def plot_pearson_hypnogram(self, con_trial, hyp_style='Block'):
        path_file = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'BM_figures', 'Block')
        M, label = BMf.cal_correlation_condition(con_trial, metric='LL', condition='Block')
        if hyp_style == 'Block':
            hypnogram = np.zeros((len(label),))
            for ix, l in enumerate(label):
                hypnogram[ix] = np.bincount(con_trial.loc[con_trial.Block == l, 'Sleep']).argmax()
            x_ax = np.arange(len(label))
            x_ax_h = np.arange(len(label))

        else:
            stimlist_hypno = pd.read_csv(os.path.join(self.path_patient_analysis, 'stimlist_hypnogram.csv'))
            stimlist_hypno = stimlist_hypno[
                np.isin(stimlist_hypno.stim_block, label) & (stimlist_hypno.sleep < 5)].reset_index(drop=True)
            hypnogram = stimlist_hypno.sleep
            x_ax_h = stimlist_hypno.ix_h  # stimlist_hypno.stim_block + (stimlist_hypno.ix_h % 1)- np.min(stimlist_hypno.stim_block)
            x_ax = stimlist_hypno.groupby(['stim_block'])['ix_h'].min().values
        x_ax_block = np.arange(len(label))
        # Create figure and subplots using gridspec
        fig = BM_plots.plot_block_hypnogram(M, hypnogram, x_ax_h, x_ax, x_ax_block)
        if hyp_style == 'Block':
            plt.savefig(os.path.join(path_file, 'Block_pearson_mean.svg'))
        else:
            plt.savefig(os.path.join(path_file, 'Block_pearson_full.svg'))
        plt.close(fig)

    def get_subnetworks(self, con_summary, parameters=['Sig', 'd', 'delay']):
        con_summary = con_summary[(con_summary.Sig > 0)].reset_index(drop=True)
        con_summary_short = con_summary[(np.isin(con_summary.Stim, np.unique(con_summary.Chan))) & (
            np.isin(con_summary.Chan, np.unique(con_summary.Stim)))].reset_index(drop=True)
        con_summary_short = con_summary_short.groupby(['Stim', 'Chan'], as_index=False)[parameters].mean().reset_index(
            drop=True)
        for par in parameters:
            # only Chan that are also stim chans
            df_pivot = con_summary_short.pivot(index='Stim', columns='Chan', values=par)
            # df_pivot = df_pivot.apply(lambda row: row.fillna(row.mean()), axis=1)
            df_pivot = df_pivot.fillna(0)
            M = df_pivot.values
            # save brain map matric format
            # initiate connectivity matrix based on total number of nodes
            n_total = len(self.labels_all)
            M_all = np.zeros((n_total, n_total)) - 1
            for sc in np.unique(
                    con_summary.Stim):  # for each selected Stim Node, get the reponse nodes (Chan) and the edge weigths (par)
                chans = con_summary.loc[con_summary.Stim == sc, 'Chan'].values
                weight = con_summary.loc[con_summary.Stim == sc, par].values
                M_all[sc, chans] = weight
            cmap = plt.get_cmap('hot')  # matplotlib.colormaps["Accent"]
            if par == 'd':
                cmap = plt.get_cmap('winter')
            elif par == 'delay':
                cmap = plt.get_cmap('autumn')
            cmap.set_bad(color='black')
            # mask some 'bad' data, in your case you would have: data == 0
            M_all = np.ma.masked_where(M_all == -1, M_all)

            fig = plt.figure(figsize=(10, 10))
            plt.suptitle(self.subj + ': BM - ' + par)
            axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            axcolor = fig.add_axes([0.9, 0.15, 0.02, 0.7])
            BM_plots.plot_BM(M_all, self.labels_all, self.hemisphere, axmatrix, axcolor=axcolor, cmap=cmap,
                             vlim=[0, np.percentile(M_all, 99)], sort=1, cat=0)
            path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'BM_figures', 'General')
            plt.savefig(os.path.join(path_output, 'BM_' + par + '.svg'))
            plt.close()

            # graphNMF.run(M, k_max = 5)

    def BM_plots_General(self, CC_summ, con_trial, reload=0):
        # todo: make it simpler
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


def start_subj(subj, cluster_method='similarity'):
    print(subj + ' -- START --')
    run_main = main(subj)
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    # load data
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    # file_CC_summ = path_patient_analysis + '\\' + folder + '\\data\\CC_summ.csv'
    file_CC_summ = path_patient_analysis + '\\' + folder + '\\data\\CC_summ_' + cluster_method + '.csv'
    summary_gen_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\summ_general.csv'  # summary_general
    # todo: make clean
    con_trial = pd.read_csv(file_con)
    # CC_summ = pd.read_csv(file_CC_summ)

    # run_main.plot_BM_sleep('NREM')
    # run_main.plot_BM_sleep('REM')
    # delay = 0
    # if delay:
    #     h5_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.h5'
    #     if os.path.isfile(h5_file):
    #         print('loading h5')
    #         EEG_resp = h5py.File(h5_file)
    #         EEG_resp = EEG_resp['EEG_resp']
    #         run_main.get_summary(con_trial, CC_summ, EEG_resp)
    wake = 1
    if wake:
        # run_main.get_summary_SS(con_trial, CC_summ)
        # run_main.get_node_features(con_trial, 1)
        run_main.connection_sleep_diff(con_trial, sig=0, skip=1)
        # run_main.BM_plots_General(CC_summ, con_trial, 0)
    # con_summary = pd.read_csv(summary_gen_path)
    # run_main.get_subnetworks(con_summary)
    blocks = 0
    if blocks:
        # Blockwise BM
        _ = run_main.save_M_block(con_trial, metrics=['LL'], savefig=1)
        # np.save(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\M_B_all.npy', M_B_all)

    print(subj + ' ----- DONE')


thread = 0
sig = 0
subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL017", "EL019", "EL020", "EL021",
         "EL022", "EL024", "EL025", "EL026", "EL027"]
# only sleep subjs
subjs = ["EL010", "EL011", "EL014", "EL015", "EL016", "EL017", "EL019", "EL020", "EL021",
         "EL022", "EL025", "EL026", "EL027"]

for subj in subjs:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'"EL021", "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016",
    if thread:
        _thread.start_new_thread(start_subj, (subj, sig))
    else:
        start_subj(subj, 'similarity')
if thread:
    while 1:
        time.sleep(1)

print('Done')
