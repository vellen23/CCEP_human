import os
import numpy as np
import mne
import imageio
import h5py
# import scipy.fftpack
import matplotlib
import pywt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# from scipy import signal
from matplotlib.colors import ListedColormap
import time
import seaborn as sns
# import scipy.io as sio
# from scipy.integrate import simps
import pandas as pd
# from scipy import fft
import matplotlib.mlab as mlab
import sys

sys.path.append('./py_functions')
import analys_func
from scipy.stats import norm
import LL_funcs
from scipy.stats import norm
from tkinter import filedialog
from tkinter import *
import ntpath
import _thread

root = Tk()
root.withdraw()
import math
import scipy
from scipy import signal
import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import platform
from glob import glob
from scipy.io import savemat

sys.path.append('./PCI/')
sys.path.append('./PCI/PCIst')

import basic_func as bf
from scipy.integrate import simps
from numpy import trapz
import IO_func as IOF
import BM_func as BMf
import tqdm
from matplotlib.patches import Rectangle
from pathlib import Path
import significance_funcs as sig_func
import freq_funcs as ff
# from tqdm.notebook import trange, tqdm
# remove some warnings
import warnings
from pathlib import Path

# I expect to see RuntimeWarnings in this block
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sleepstate_labels = ['NREM', 'REM', 'Wake']


class main:
    def __init__(self, subj):
        #  basics, get 4s of data for each stimulation, [-2,2]s
        self.folder = 'BrainMapping'
        self.cond_folder = 'CR'
        self.path_patient_analysis = 'T:\EL_experiment\Projects\EL_experiment\Analysis\Patients\\' + subj
        self.path_patient = 'T:\EL_experiment\Patients\\' + subj + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj

        self.Fs = 500
        self.dur = np.zeros((1, 2), dtype=np.int32)
        self.dur[0, :] = [-1, 3]
        self.dur_tot = np.int32(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        # load patient specific information
        lbls = pd.read_excel(os.path.join(path_patient, 'infos', subj + "_labels.xlsx"), header=0, sheet_name='BP')
        self.labels_all = lbls.label.values
        self.labels_C = lbls.Clinic.values
        stimlist = pd.read_csv(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\stimlist_' + self.cond_folder + '.csv')
        #
        labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
            stimlist,
            lbls)
        self.labels_region_L = lbls.Hemisphere.values + '_' + labels_region
        self.subj = subj
        self.labels_region = labels_region

        # regions information
        self.CR_color = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\Analysis\BrainMapping\CR_color.xlsx",
                                      header=0)
        regions = pd.read_excel("Y:\eLab\Projects\EL_experiment\Analysis\Patients\Across\elab_labels.xlsx",
                                sheet_name='regions',
                                header=0)
        self.color_regions = regions.color.values
        self.regions = regions

        # C = regions.label.values
        # self.path_patient   = path_patient
        # self.path_patient_analysis = os.path.join(os.path.dirname(os.path.dirname(self.path_patient)), 'Projects\EL_experiment\Analysis\Patients', subj)
        ##bad channels
        non_stim = np.arange(len(self.labels_all))
        non_stim = np.delete(non_stim, StimChanIx, 0)
        WM_chans = np.where(self.labels_region == 'WM')[0]
        self.bad_all = np.unique(np.concatenate([WM_chans, bad_region, bad_chans, non_stim])).astype('int')
        stim_chans = np.arange(len(labels_all))
        self.stim_chans = np.delete(stim_chans, self.bad_all, 0)
        Path(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures').mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\LL').mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\Prob').mkdir(
            parents=True, exist_ok=True)

    def plot_BM_CR_block(self, M, areas, type_label, t, method, save=1):
        # either probability or LL
        M[np.isnan(M)] = 0
        time = str(t).zfill(2) + ':00'
        fig = plt.figure(figsize=(15, 15))
        vmin = 0
        vmax = 1
        if method == 'LL':
            vmin = 1
            vmax = np.min([15, np.nanpercentile(M, 95)])
        axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])  # x, y, (start posiion), lenx, leny
        im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
        plt.suptitle(type_label + ', ' + time + '-- ' + method)
        plt.xlim([-1.5, len(self.labels) - 0.5])
        plt.ylim([-0.5, len(self.labels) + 0.5])
        plt.xticks(range(len(self.labels)), self.labels, rotation=90)
        plt.yticks(range(len(self.labels)), self.labels)
        for i in range(len(self.labels)):
            r = areas[i]
            axmatrix.add_patch(Rectangle((i - 0.5, len(self.labels) - 0.5), 1, 1, alpha=1,
                                         facecolor=self.color_regions[np.where(self.regions == r)[0][0]]))
            axmatrix.add_patch(
                Rectangle((-1.5, i - 0.5), 1, 1, alpha=1,
                          facecolor=self.color_regions[np.where(self.regions == r)[0][0]]))
        # Plot colorbar.
        axcolor = fig.add_axes([0.04, 0.85, 0.08, 0.08])  # x, y, x_len, y_len
        circle1 = plt.Circle((0.5, 0.5), 0.4, color=self.CR_color.c[t], alpha=self.CR_color.a[t])
        plt.text(0.3, 0.3, time)
        plt.axis('off')
        axcolor.add_patch(circle1)
        axcolor = fig.add_axes([0.9, 0.15, 0.01, 0.7])  # x, y, x_len, y_len
        plt.colorbar(im, cax=axcolor)

        if save:
            plt.savefig(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures\\BM_' + method + '\\BM_' + type_label + '.jpg')
            # plt.savefig(path_patient_analysis+'\\' + folder + '\\' + cond_folder +'\\figures/BM_LL\\BM_'+label+'.jpg')
            plt.close(fig)  # plt.show()#
        else:
            plt.show()

    def get_sig(self, sc, rc, con_trial, M_GT, t_resp, sig_mean, EEG_CR, p=95, exp=2, w_cluster=0.25):
        # for each trial get significance level based on surrogate (Pearson^2 * LL)
        dat = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1)]
        EEG_trials = ff.lp_filter(EEG_CR[[[rc]], dat.Num.values.astype('int'), :], 45, self.Fs)
        LL_trials = LLf.get_LL_all(EEG_trials, self.Fs, w_cluster, 1, 0)
        # surr
        pear_surr_all = []
        f = 1
        for t_test in [0.125, 0.3, 0.5, 0.7, 1.5, 1.8, 2.2, 2.6]:  # surrogates times, todo: in future blockwise
            s = (-1) ** f
            pear = get_pearson2mean(M_GT[1, :], s * EEG_trials[0], tx=t_0 + t_resp, ty=t_test, win=w_cluster,
                                    Fs=500)  # Pearson# Pearson
            pear2 = get_pearson2mean(M_GT[2, :], s * EEG_trials[0], tx=t_0 + t_resp, ty=t_test, win=w_cluster,
                                     Fs=500)  # Pearson# Pearson
            LL = LL_trials[0, :, int(t_test + w_cluster / 2) * self.Fs]
            # pear_surr = np.arctanh(np.max([pear,pear2],0))*LL
            pear_surr = np.sign(np.max([pear, pear2], 0)) * abs(np.max([pear, pear2], 0) ** exp) * np.sqrt(LL)
            pear_surr_all = np.concatenate([pear_surr_all, pear_surr])
            f = f + 1

        # real
        t_test = t_0 + t_resp
        pear = get_pearson2mean(M_GT[1, :], EEG_trials[0], tx=t_0 + t_resp, ty=t_test, win=w_cluster,
                                Fs=500)  # Pearson# Pearson
        pear2 = get_pearson2mean(M_GT[2, :], EEG_trials[0], tx=t_0 + t_resp, ty=t_test, win=w_cluster,
                                 Fs=500)  # Pearson# Pearson
        pear3 = get_pearson2mean(M_GT[0, :], EEG_trials[0], tx=t_0 + t_resp, ty=t_test, win=w_cluster,
                                 Fs=500)  # Pearson# Pearson
        LL = LL_trials[0, :, int(t_test + w_cluster / 2) * self.Fs]
        pear = np.sign(np.max([pear, pear2, pear3], 0)) * abs(np.max([pear, pear2, pear3], 0) ** exp) * np.sqrt(LL)
        sig = (pear > np.nanpercentile(pear_surr_all, p)) * 1
        con_trial.loc[
            (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1), 'Sig'] = sig * sig_mean
        return con_trial

    def save_M_block(self, con_trial, metric='LL', savefig=1):
        # labels:
        labels_sel = np.delete(self.labels_all, self.bad_all, 0)
        areas_sel = np.delete(self.labels_region_L, self.bad_all, 0)
        # sort
        ind = np.argsort(areas_sel)
        areas_sel = np.delete(self.labels_region, self.bad_all, 0)
        labels_sel = labels_sel[ind]
        areas_sel = areas_sel[ind]
        # blocks
        M_B_all = np.zeros((int(np.max(con_trial.Block) + 1), len(self.labels_all), len(self.labels_all)))
        for b in np.unique(con_trial.Block).astype('int'):
            if metric == 'LL':
                summ = con_trial[(con_trial.Block == b) & (con_trial.Artefact < 1)]  # (con_trial.Sleep==s)&

            else:  # probability
                summ = con_trial[
                    (con_trial.Sig == 1) & (con_trial.Block == b) & (con_trial.Artefact < 1)]  # (con_trial.Sleep==s)&

            summ = summ.groupby(['Stim', 'Chan'], as_index=False)[[metric, 'd']].mean()
            # summ = summ[summ.Sig==1]
            t = np.bincount(con_trial.loc[con_trial.Block == b, 'Hour']).argmax()
            M = np.zeros((len(self.labels_all), len(self.labels_all)))
            M[:, :] = np.nan
            for sc in np.unique(summ.Stim).astype('int'):
                chan = summ.loc[summ.Stim == sc, 'Chan'].values.astype('int')
                LL = summ.loc[summ.Stim == sc, metric].values
                M[sc, chan] = LL

            M_B_all[b, :, :] = M
            if savefig:
                M_resp = np.delete(np.delete(M, self.bad_all, 0), self.bad_all, 1)
                M_resp = M_resp[ind, :]
                M_resp = M_resp[:, ind]
                ll = 'BM' + str(int(b)).zfill(2)  # +', '+sleep_states[s]
                plot_BM_CR_trial_sig(M_resp, labels_sel, areas_sel, ll, t, metric, savefig)
        # pearson correlation across all blocks
        M_B_allp = np.zeros((int(np.max(con_trial.Block) + 1), len(self.labels_all), len(self.labels_all)))
        M_B_allp[:, :, :] = M_B_all[:, :, :]
        M_B_allp[np.isnan(M_B_allp)] = 0
        M_B_pear = np.zeros((int(np.max(con_trial.Block) + 1), int(np.max(con_trial.Block) + 1)))
        for b1 in np.unique(con_trial.Block).astype('int'):
            for b2 in np.unique(con_trial.Block).astype('int'):
                M_B_pear[b1, b2] = np.corrcoef(M_B_allp[b1, :, :].flatten(), M_B_allp[b2, :, :].flatten())[0, 1]
        plt.figure(figsize=(8, 8))
        plt.pcolor(M_B_pear[:, :], cmap='hot', vmin=np.percentile(M_B_pear[:, :], 10),
                   vmax=np.percentile(M_B_pear[:, :], 95))
        plt.savefig(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\figures\\BM_' + method + '\\BM_corr.jpg')
        plt.savefig(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\figures\\BM_' + method + '\\BM_corr.svg')
        plt.close(fig)  # plt.show()#

        return M_B_all

    def save_M_sleep(self, con_trial, metric='LL', savefig=1):
        # labels:
        labels_sel = np.delete(self.labels_all, self.bad_all, 0)
        areas_sel = np.delete(self.labels_region_L, self.bad_all, 0)
        # sort
        ind = np.argsort(areas_sel)
        areas_sel = np.delete(self.labels_region, self.bad_all, 0)
        labels_sel = labels_sel[ind]
        areas_sel = areas_sel[ind]
        # blocks
        M_B_all = np.zeros((len(sleepstate_labels), len(self.labels_all), len(self.labels_all)))
        for ss, b in zip(sleepstate_labels, np.arange(len(sleepstate_labels))):
            if metric == 'LL':
                summ = con_trial[(con_trial.SleepState == ss) & (con_trial.Artefact < 1)]  # (con_trial.Sleep==s)&
            else:  # probability
                summ = con_trial[
                    (con_trial.Sig == 1) & (con_trial.SleepState == ss) & (
                                con_trial.Artefact < 1)]  # (con_trial.Sleep==s)&

            summ = summ.groupby(['Stim', 'Chan'], as_index=False)[[metric, 'd']].mean()
            # summ = summ[summ.Sig==1]
            t = np.bincount(con_trial.loc[con_trial.Block == b, 'Hour']).argmax()
            M = np.zeros((len(self.labels_all), len(self.labels_all)))
            M[:, :] = np.nan
            for sc in np.unique(summ.Stim).astype('int'):
                chan = summ.loc[summ.Stim == sc, 'Chan'].values.astype('int')
                LL = summ.loc[summ.Stim == sc, metric].values
                M[sc, chan] = LL

            M_B_all[b, :, :] = M
            if savefig:
                M_resp = np.delete(np.delete(M, self.bad_all, 0), self.bad_all, 1)
                M_resp = M_resp[ind, :]
                M_resp = M_resp[:, ind]
                plot_BM_CR_trial_sig(M_resp, labels_sel, areas_sel, ss, t, metric, savefig)

        return M_B_all
