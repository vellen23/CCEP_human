import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import seaborn as sns
import pandas as pd
import sys

sys.path.append('T:\\EL_experiment\\Codes\\UBELIX_EvM\\NMF\\functions')
import NMF_funcs
sys.path.append('T:\\EL_experiment\\Codes\\CCEP_human\\Python_Analysis\\py_functions')

from scipy.stats import norm
from tkinter import *
import _thread
import h5py


root = Tk()
root.withdraw()
import basic_func as bf
import BM_stats
import tqdm
import matplotlib
from matplotlib.patches import Rectangle
import warnings
from pathlib import Path

import BM_plots

# I expect to see RuntimeWarnings in this block
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sleepstate_labels = ['NREM', 'REM', 'Wake']

folder = 'BrainMapping'
cond_folder = 'CR'
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

conditions = ['S_V', 'S_A', 'C_A', 'C_V']


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
        self.hemisphere = lbls.Hemisphere
        atlas_regions = pd.read_excel(
            "X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients\\Across\\elab_labels.xlsx",
            sheet_name="atlas")
        self.labels_region = labels_region
        for i in range(len(labels_all)):
            area_sel = " ".join(re.findall("[a-zA-Z_]+", labels_all[i]))
            self.labels_region[i] = atlas_regions.loc[atlas_regions.Abbreviation == area_sel, "Region"].values[0]

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
        Path(self.path_patient_analysis + '\\' + folder + '\\' + "CT\\").mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + folder + '\\' + "CT\\data\\").mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + folder + '\\' + "CT\\figures\\").mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + folder + '\\' + "CT\\NMF\\").mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + folder + '\\' + "CT\\NMF\\figures\\").mkdir(
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

    def load(self):
        # Construct file paths
        file_con = os.path.join(self.path_patient_analysis, folder, cond_folder, "data", "con_trial_all.csv")
        h5_file = os.path.join(self.path_patient_analysis, folder, cond_folder, "data", f'EEG_{cond_folder}.h5')
        # Load con_trial DataFrame
        con_trial = pd.read_csv(file_con)
        # Convert specific columns to int
        int_columns = ["Num", "Stim", "Chan", "Artefact", "Num_block", "Date", "Block", "Sleep", "Hour"]
        con_trial[int_columns] = con_trial[int_columns].astype(int)
        # CT mapping based on subjects
        ct_mapping = {
            "EL022": {18: "S_V", 19: "S_A", 20: "C_V", 21: "C_A"},
            "EL025": {32: "S_V", 33: "S_A", 34: "C_V", 35: "C_A"},
            "EL024": {36: "S_V", 37: "S_A"}
        }
        con_trial["CT"] = con_trial.apply(lambda row: ct_mapping.get(self.subj, {}).get(row["Block"], "BL"), axis=1)
        # Filter and process con_trial_CT
        con_trial_CT = con_trial[(con_trial.Artefact < 1) & (con_trial.Sig > -1) & (con_trial.Sleep == 0) & (
                (con_trial.Hour > 10) | (con_trial.Hour < 20))].reset_index(drop=True)
        con_trial_CT.insert(4, 'LL_sig', con_trial_CT.LL * con_trial_CT.Sig)
        # Process con_mean_CT
        con_mean_CT = con_trial_CT.groupby(['Stim', 'Chan', 'CT'], as_index=False)[['Sig', 'LL', 'LL_sig']].mean()
        con_mean_CT['Con_ID'] = con_mean_CT.groupby(['Stim', 'Chan']).ngroup().reset_index(drop=True)
        # Calculate LL_norm
        ll_bl_series = con_mean_CT[con_mean_CT['CT'] == 'BL'].groupby('Con_ID')['LL_sig'].first()
        con_mean_CT['LL_norm'] = con_mean_CT['LL'] / con_mean_CT['Con_ID'].map(ll_bl_series)
        # Load EEG_resp from h5 file if it exists
        if os.path.isfile(h5_file):
            EEG_resp = h5py.File(h5_file)
            self.EEG_resp = EEG_resp['EEG_resp']
        # Keep main tables in self
        self.con_trial_CT = con_trial_CT
        self.con_mean_CT = con_mean_CT

    def BM_LL_ratio(self, metric="LL_WOI"):
        path_fig = os.path.join(self.path_patient_analysis, folder, "CT", "figures", "LL_V_A" + metric + ".jpg")
        con_trial_BL = self.con_trial_CT[np.isin(self.con_trial_CT, conditions)]
        con_trial_BL = con_trial_BL.reset_index(drop=True)
        con_trial_BL["CT"].apply(lambda ct: ct[2])

        # Map CT values to CT2 values (A or V)
        con_trial_BL["CT2"] = con_trial_BL["CT"].apply(lambda ct: ct[2])
        # Calculate mean LL for each connection in CT2 == A and CT2 == V
        grouped = con_trial_BL.groupby(["Stim", "Chan", "CT2"])[["Sig", metric]].mean().reset_index()
        grouped = grouped[grouped.Sig > 0.2].reset_index()
        # Pivot the table to get A and V values in separate columns
        pivot_table = grouped.pivot_table(index=["Stim", "Chan"], columns="CT2", values=metric).reset_index()

        # Calculate the LL_ratio based on the formula
        pivot_table["LL_r"] = (1 - (pivot_table[["A", "V"]].min(axis=1) / pivot_table[["A", "V"]].max(axis=1)))
        # Adding direction
        pivot_table.loc[(pivot_table.A > pivot_table.V), "LL_r"] = - pivot_table.loc[
            (pivot_table.A > pivot_table.V), "LL_r"]

        matrix = np.full((np.max([pivot_table.Chan.max(), pivot_table.Stim.max()]) + 1,
                          np.max([pivot_table.Chan.max(), pivot_table.Stim.max()]) + 1), np.nan)  # Initialize with NaN

        # Fill the matrix with LL_norm values
        for index, row in pivot_table.iterrows():
            matrix[row["Stim"].astype('int'), row["Chan"].astype('int')] = row["LL_r"]

        cmap = matplotlib.colormaps["RdBu_r"]
        cmap.set_bad(color='black')

        fig = plt.figure(figsize=(10, 10))
        axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        axcolor = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        vlim = [-0.5, 0.5]
        BM_plots.plot_BM(matrix, self.labels_all, self.hemisphere, axmatrix, axcolor=axcolor, cmap=cmap, vlim=vlim,
                         sort=1)
        axcolor.set_ylabel('LL ratio')
        axcolor.set_yticks([vlim[0], vlim[0] / 2, 0, vlim[1] / 2, vlim[1]])
        axcolor.set_yticklabels(["A>V", str(vlim[0] / 2), " 0", str(vlim[1] / 2), "V>A"])
        plt.suptitle(subj + ' -- LL ratio of auditory vs visuell focus', y=0.9)
        plt.savefig(path_fig, dpi=300)

    def plot_BM_BL_change(self):
        path_fig = os.path.join(self.path_patient_analysis, folder, "CT", "figures", "LL_BL_change.jpg")
        cmap = matplotlib.colormaps["RdBu_r"]
        cmap.set_bad(color='grey')

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))

        for row_ix, start in enumerate(['S', 'C']):
            for col_ix, end in enumerate(['A', 'V']):
                ax = axs[row_ix, col_ix]

                cond = start + '_' + end
                df = self.con_mean_CT[(self.con_mean_CT.CT == cond)]
                df = df.reset_index(drop=True)
                df.Stim = df.Stim.astype('int')
                df.Chan = df.Chan.astype('int')
                if len(df) > 0:
                    matrix = np.full(
                        (np.max([df.Chan.max(), df.Stim.max()]) + 1, np.max([df.Chan.max(), df.Stim.max()]) + 1),
                        np.nan)  # Initialize with NaN

                    # Fill the matrix with LL_norm values
                    for index, row in df.iterrows():
                        matrix[row["Stim"], row["Chan"]] = row["LL_norm"]

                    BM_plots.plot_BM(matrix, self.labels_all, self.hemisphere, ax, axcolor=None, cmap=cmap,
                                     vlim=[0.7, 1.3], sort=1)
                    # remove x-ticks and y-ticks if not
                    if row_ix + col_ix > 0:
                        ax.set_yticks([])
                    else:
                        ax.set_xticks([])
                    if col_ix == 1:
                        ax.set_xticks([])
        plt.tight_layout()
        plt.savefig(path_fig, dpi=300)

    def NMF_trials(self, metric='LL_WOI', k0=3, k1=10):
        """
            Perform NMF analysis on connection trial data and generate H_table and Con_table.

            Parameters:
                con_trial_CT (pd.DataFrame): DataFrame containing connection trial data.
                conditions (list): List of conditions to consider.
                metric (str): Column name for the metric to be used.
                k0 (int): Minimum number of clusters for NMF.
                k1 (int): Maximum number of clusters for NMF.

            Returns:
                W (np.ndarray): Basis function matrix.
                H (np.ndarray): Activation function matrix.
                H_table (pd.DataFrame): DataFrame containing H coefficients for each trial and cluster.
                Con_table (pd.DataFrame): DataFrame containing connection information and cluster assignments.
            """
        n_node = len(self.labels_all)
        # remove BL trials
        con_trial = self.con_trial_CT[np.isin(self.con_trial_CT.CT, conditions)].reset_index(drop=True)
        con_trial["CT"].apply(lambda ct: ct[2])  # pool auditory and visuell together
        # unique Connection ID
        con_trial['Con_ID'] = con_trial.groupby(['Stim', 'Chan']).ngroup()

        # add n_trial label (1-3) for each condition
        # Calculate cumulative counts for each group
        con_trial['n_trial'] = con_trial.groupby(['Stim', 'Chan', 'CT']).cumcount()
        for i, ct in enumerate(np.unique(con_trial.CT)):
            shift = np.max(con_trial.n_trial) + 1
            con_trial.loc[con_trial.CT == ct, 'n_trial'] = con_trial.loc[
                                                               con_trial.CT == ct, 'n_trial'] + shift
        con_trial.n_trial = con_trial.n_trial - np.min(con_trial.n_trial)
        ## normalize metric based on the mean (within conenction ID)
        con_trial.loc[con_trial[metric] == -1, metric] = np.nan
        con_trial['norm'] = con_trial.groupby('Con_ID').apply(
            lambda x: x[metric] / x[metric].mean()).reset_index(0, drop=True)
        ## fill nan, first with the mean - if still nan - put to 0
        con_trial['norm'].fillna(con_trial.groupby('Con_ID')['norm'].transform('mean'), inplace=True)
        con_trial['norm'].fillna(0, inplace=True)

        # get input matrix V
        V = con_trial.pivot_table(index=["Con_ID"], columns="n_trial", values='norm')
        V = V.values
        V[np.isnan(V)] = np.nanmean(V)
        k1 = np.min([k1, np.min(V.shape) - 1])
        _, instability = NMF_funcs.stabNMF(V, num_it=20, k0=k0, k1=k1, init='nndsvda', it=2000)
        # select rank with lowest instability value
        ranks = np.arange(k0, k1 + 1)
        k = ranks[np.argmin(instability)]

        # rerun NMF with chosen best rank
        print('running NMF with a chosen rank of ' + str(k))
        W, H = NMF_funcs.get_nnmf(V, k, init='nndsvda', it=2000)
        clusters = NMF_funcs.get_clusters(W)

        # Create H_table
        H_arr = np.zeros((H.shape[0] * H.shape[1], 3))
        for i in range(H.shape[1]):
            H_arr[i * k:(i + 1) * k, 0] = i  # n_trial
            for j in range(k):  # for each cluster
                H_arr[(i * k) + j, 1] = H[j, i]
                H_arr[(i * k) + j, 2] = j
        H_table = pd.DataFrame(H_arr, columns=['n_trial', 'H', 'C'])

        # Merge CT information from con_trial
        H_table = H_table.merge(
            con_trial.groupby(['n_trial', 'CT'], as_index=False)['LL'].mean(), on=['n_trial']
        )

        con_clusters = np.zeros((W.shape[0], 2))
        con_clusters[:, 0] = np.arange(W.shape[0])
        con_clusters[:, 1] = np.argmax(W, 1)

        Con_table = pd.DataFrame(con_clusters, columns=['Con_ID', 'C'])
        Con_table = Con_table.merge(
            con_trial[['Con_ID', 'Stim', 'Chan']].drop_duplicates(),
            on='Con_ID',
            how='left'
        )
        # plot
        con_mean = self.con_mean_CT.groupby(['Con_ID', 'Stim', 'Chan'], as_index=False)['Sig'].mean()
        con_mean = con_mean.merge(Con_table)
        matrix = np.full((n_node, n_node), np.nan)  # Initialize with NaN

        # Fill the matrix with LL_norm values
        for index, row in con_mean[con_mean.Sig > 0].iterrows():
            matrix[row["Stim"].astype('int'), row["Chan"].astype('int').astype('int')] = row['C'].astype('int')

        cmap = matplotlib.colormaps["Accent"]
        cmap.set_bad(color='black')

        path_fig = os.path.join(self.path_patient_analysis, folder, "CT", "NMF", "figures", "NMF_BM_" + metric + ".jpg")

        fig = plt.figure(figsize=(10, 10))
        axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        # axcolor = fig.add_axes([0.9,0.15,0.02,0.7])
        vlim = [-0.5, 0.5]
        BM_plots.plot_BM(matrix, self.labels_all, self.hemisphere, axmatrix, axcolor=None, cmap=cmap,
                         vlim=[np.nanmin(matrix), np.nanmax(matrix)], sort=1)

        plt.suptitle(subj + ' -- LL ratio of auditory vs visuell focus', y=0.9)
        plt.savefig(path_fig, dpi=300)
        plt.show()

        path_fig = os.path.join(self.path_patient_analysis, folder, "CT", "NMF", "figures",
                                "NMF_H_boxplot_C_" + metric + ".jpg")
        sns.boxenplot(x='C', y='H', hue='CT', data=H_table)
        plt.savefig(path_fig, dpi=300)
        plt.show()

        path_fig = os.path.join(self.path_patient_analysis, folder, "CT", "NMF", "figures",
                                "NMF_H_boxplot_CT_" + metric + ".jpg")
        sns.boxplot(x='CT', y='H', hue='C', data=H_table)
        plt.savefig(path_fig, dpi=300)
        plt.show()

        return W, H, H_table, Con_table

        # run NMF


def start_subj(subj, cluster_method='similarity', sig=0):
    print(subj + ' -- START --')
    run_main = main(subj)
    run_main.load()
    run_main.NMF_trials(metric='LL_WOI', k0=3, k1=10)
    run_main.BM_LL_ratio(metric="LL_WOI")
    run_main.plot_BM_BL_change()

    print(subj + ' ----- DONE')


thread = 0
sig = 0
# # # todo: 'EL009',
for subj in ['EL024','EL022',
             'EL025']:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'"EL021", "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016",
    if thread:
        _thread.start_new_thread(start_subj, (subj, sig))
    else:
        start_subj(subj, 'similarity', 0)
if thread:
    while 1:
        time.sleep(1)
