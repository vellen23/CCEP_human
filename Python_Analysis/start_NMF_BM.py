import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import sys
import json

sys.path.append('T:\EL_experiment\Codes\CCEP_human\Python_Analysis\py_functions')
sys.path.append('X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram')
from main_script import ConnectogramPlotter
from scipy.stats import norm
from tkinter import *
import _thread
import h5py

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
import matplotlib.dates as mdates

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

path_connectogram = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram'


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
        bad_region = np.where((labels_region == 'Unknown') | (labels_region == 'WM') | (labels_region == 'OUT') | (
                labels_region == 'Putamen'))[0]
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

        # labels:
        labels_sel = np.delete(self.labels_all, self.bad_all, 0)
        areas_sel = np.delete(self.labels_region_L, self.bad_all, 0)
        # sort
        ind = np.argsort(areas_sel)
        areas_sel = np.delete(self.labels_region, self.bad_all, 0)
        self.labels_sel = labels_sel[ind]
        self.areas_sel = areas_sel[ind]
        self.ind = ind

    def load_data(self):
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF')
        self.con_trial = pd.read_csv(os.path.join(path_output, self.subj + '_con_trial_cluster.csv'))
        with h5py.File(os.path.join(path_output, self.subj + '_con_trial_nmf.h5'), 'r') as hf:
            # Read W and H datasets
            self.W = hf['W'][:]
            self.H = hf['H'][:]

        # Read cluster assignments from the JSON file
        with open(os.path.join(path_output, self.subj + '_con_trial_nmf_cluster.json'), 'r') as json_file:
            self.column_cluster_assignments = json.load(json_file)

        # sleep data
        self.stimlist_sleep = pd.read_csv(os.path.join(self.path_patient_analysis, 'stimlist_hypnogram.csv'))

    # Format x-axis tick labels as daytime hours
    def format_time_hour(self, x, pos):
        while x > 24:
            x -= 24
        return f'{int(x):02d}:00'

    def plot_H_hypnogram(self):
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'figures')
        os.makedirs(path_output, exist_ok=True)  # Create directories if they don't exist
        # adding "ic_chron" which adds +24h if there is a new day to keep it chronological
        for d in range(len(np.unique(self.stimlist_sleep.date))):
            self.stimlist_sleep.loc[self.stimlist_sleep.date == np.unique(self.stimlist_sleep.date)[d], 'ix_chron'] = \
                self.stimlist_sleep.loc[
                    self.stimlist_sleep.date == np.unique(self.stimlist_sleep.date)[d], 'ix_h'] + d * 24
        stimlist_hypno = self.stimlist_sleep
        stimlist_hypno.loc[stimlist_hypno.sleep > 4, 'sleep'] = 0  # everythin that's greter than 4 is also wake
        # some calculations to have the same x time axis for both plots

        rk = self.H.shape[0]
        n_block = self.H.shape[1]
        blocks_all = np.unique(self.con_trial.Block)
        # Generating x-axis values
        x_ax_block = np.arange(0, n_block).astype('float')
        for ix_b, b in enumerate(np.unique(self.con_trial.Block)):
            x_ax_block[ix_b] = np.mean(self.stimlist_sleep.loc[self.stimlist_sleep.stim_block == b, 'ix_chron'])
        # x_ticks_h = np.arange(0, np.max(blocks_all) + 1, step=5)
        # labels_hour = [f'{int(h):02d}:00' for h in np.floor(self.stimlist_sleep.ix_h[x_ticks_h])]
        timeline = np.ceil(np.max(self.stimlist_sleep.ix_chron) - np.min(self.stimlist_sleep.ix_chron)).astype('int')
        # Create figure and subplots
        fig, (ax_h, ax) = plt.subplots(2, 1, figsize=(timeline / 3.5, 7))

        # Plot hypnogram
        ax_h.plot(self.stimlist_sleep.ix_chron, self.stimlist_sleep.sleep, c='black', linewidth=2)
        ax_h.axhspan(-1, 0.2, color=color_elab[0, :])
        ax_h.fill_between(self.stimlist_sleep.ix_chron, self.stimlist_sleep.sleep, -1, color=color_elab[0, :])
        ax_h.set_yticks([0, 1, 2, 3, 4])
        ax_h.set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
        ax_h.set_ylim([-1, 5])
        ax_h.invert_yaxis()
        ax_h.set_xticks([])
        ax_h.set_ylabel('Score', fontsize=15)

        # Plot H coefficients
        for i in range(rk):
            ax.plot(x_ax_block, self.H[i], linewidth=4, label=f'H{i + 1}')
        # ax.set_xticks(x_ticks_h)
        # ax.set_xticklabels(labels_hour, fontsize=12)
        ax.set_ylabel('H Coefficients', fontsize=15)
        ax.set_xlabel('Time', fontsize=15)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_time_hour))
        # same xlim as ax_h to be perfectly aligned
        ax.set_xlim(ax_h.get_xlim())

        plt.tight_layout()
        plt.savefig(os.path.join(path_output, 'H_hypnogram.svg'))

    def plot_clusters_connectogram(self):
        cwp = os.getcwd()

        plotter = ConnectogramPlotter()
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'cluster_connectogram')
        os.makedirs(path_output, exist_ok=True)  # Create directories if they don't exist

        con_table = self.con_trial.groupby(['Con_ID', 'Chan', 'Stim'], as_index=False)[['LL', 'Sig', 'd']].mean()
        con_table.Chan = con_table.Chan.astype('int')
        con_table.Stim = con_table.Stim.astype('int')
        con_table.insert(0, 'Subj', self.subj)
        # Create new columns 'StimA' and 'ChanA' using vectorized operations
        con_table['StimA'] = [re.sub(r'\d', '', label) for label in self.labels_all[con_table['Stim']]]
        con_table['ChanA'] = [re.sub(r'\d', '', label) for label in self.labels_all[con_table['Chan']]]

        os.chdir(path_connectogram)

        n_cluster = len(self.column_cluster_assignments)
        for ix_cluster, cluster in enumerate(self.column_cluster_assignments):
            cluster_chans = self.column_cluster_assignments[cluster]
            con_sel = con_table.loc[np.isin(con_table.Con_ID, cluster_chans)]
            con_sel = con_sel.reset_index(drop=True)
            path_save = os.path.join(path_output, 'Cluster_' + cluster + '.jpg')
            # plot connectogram and save figure
            plotter.load_data(con_sel, path_save)
            plotter.show_plot(subj + ' Cluster ' + str(ix_cluster + 1))

        os.chdir(cwp)

    def plot_LL_clusters(self):
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'figures')
        os.makedirs(path_output, exist_ok=True)  # Create directories if they don't exist
        con_trial = self.con_trial[self.con_trial.SleepState != 'NREM1']
        # remove outliers
        thr = np.percentile(con_trial.LL_norm, 99)
        con_trial = con_trial[(con_trial.LL_norm < thr)]
        con_trial = con_trial.reset_index(drop=True)
        n_cluster = len(self.column_cluster_assignments)
        fig, axes = plt.subplots(1, n_cluster, figsize=(n_cluster * 4, 6), sharey=True)
        for ix_cluster, cluster in enumerate(self.column_cluster_assignments):
            cluster_chans = self.column_cluster_assignments[cluster]
            con_sel = con_trial.loc[np.isin(con_trial.Con_ID, cluster_chans)]
            con_sel = con_sel.reset_index(drop=True)

            ax = sns.boxplot(x='SleepState', y='LL_norm', data=con_sel, ax=axes[ix_cluster])
            ax.set_title(f'Cluster {ix_cluster + 1}' + ' - #Connections: ' + str(len(cluster_chans)))
            ax.set_xlabel('Sleep State')
            ax.set_ylabel('normalized LL ')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.savefig(os.path.join(path_output, 'Cluster_SleepState.jpg'))
        plt.show()
        print('stop')


def start_subj(subj):
    print(subj + ' -- START --')
    run_main = main(subj)
    run_main.load_data()
    run_main.plot_H_hypnogram()
    run_main.plot_clusters_connectogram()
    run_main.plot_LL_clusters()

    print(subj + ' ----- DONE')


thread = 0
subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL017", "EL019", "EL020",
         "EL021", "EL022", "EL025", "EL026"]

subjs = ["EL024"]
for subj in subjs:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'"EL021", "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016",
    if thread:
        _thread.start_new_thread(start_subj, (subjs))
    else:
        start_subj(subj)
if thread:
    while 1:
        time.sleep(1)
