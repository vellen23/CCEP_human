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

sys.path.append('T:\\EL_experiment\Codes\CCEP_human\Python_Analysis\py_functions')
sys.path.append('X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram')
sys.path.append('T:\\EL_experiment\Codes\\UBELIX_EvM\\NMF\\functions')
from main_script import ConnectogramPlotter
from scipy.stats import norm
import BM_CR_funcs as BMf_cluster
from tkinter import *
import _thread
import h5py
from matplotlib import gridspec

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
from sklearn.cluster import KMeans
import BM_plots
import BM_func
from scipy.cluster.hierarchy import linkage, cophenet

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

    def get_clusters(self, W):
        column_cluster_assignments = {}
        num_cluster = W.shape[1]

        for cluster_idx in range(num_cluster):
            W_values = W[:, cluster_idx].reshape(-1, 1)  # Reshape the values for clustering
            kmeans = KMeans(n_clusters=2).fit(W_values)
            higher_value_cluster = np.argmax(kmeans.cluster_centers_)

            for channel, assignment in enumerate(kmeans.labels_):
                if assignment == higher_value_cluster:
                    column_cluster_assignments.setdefault(cluster_idx, []).append(channel)

        return column_cluster_assignments

    def load_data(self, conNMF=1):
        self.sig_thr = 0.1
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF')
        con_trial = pd.read_csv(
            os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'data', 'con_trial_all.csv'))
        # self.con_trial = pd.read_csv(os.path.join(path_output, self.subj + '_con_trial_cluster.csv'))
        if conNMF:
            path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'conNMF')
            metric = pd.read_csv(os.path.join(path_output, 'metrics.csv'))
            # todo: define what is best k
            k = metric.loc[np.argmax(metric[
                                         'delta_k (AUC)']), 'Rank']  # metric.loc[(metric['Instability index'] == np.min(metric['Instability index'])), 'Rank'].values[0]
            self.H = pd.read_csv(os.path.join(path_output, 'k=' + str(k), 'H_best.csv'), header=None).values
            self.W = pd.read_csv(os.path.join(path_output, 'k=' + str(k), 'W_best.csv'), header=None).values

            # self.column_cluster_assignments = self.get_clusters(self.W)

        else:
            with h5py.File(os.path.join(path_output, self.subj + '_con_trial_nmf.h5'), 'r') as hf:
                # Read W and H datasets
                self.W = hf['W'][:]
                self.H = hf['H'][:]

            # Read cluster assignments from the JSON file
            with open(os.path.join(path_output, self.subj + '_con_trial_nmf_cluster.json'), 'r') as json_file:
                self.column_cluster_assignments = json.load(json_file)
        self.V, self.con_trial = BMf_cluster.get_V(con_trial)
        data_plot = self.con_trial.groupby(['Con_ID', 'Stim', 'Chan', 'Block'], as_index=False)[
            ['LL', 'Sig']].mean().reset_index(drop=True)
        sig_con = data_plot.groupby(['Con_ID'], as_index=False)['Sig'].mean()
        sig_con = np.unique(sig_con.loc[(sig_con.Sig > self.sig_thr), 'Con_ID'])
        # sleep data
        self.stimlist_sleep = pd.read_csv(os.path.join(self.path_patient_analysis, 'stimlist_hypnogram.csv'))
        ## adding cluster
        con_clusters = np.zeros((self.W.shape[0], 2))
        con_clusters[:, 0] = np.unique(self.con_trial.loc[np.isin(self.con_trial.Con_ID, sig_con), 'Con_ID'])
        con_clusters[:, 1] = NMFf.assgin_cluster(self.V, self.H)

        Con_table = pd.DataFrame(con_clusters, columns=['Con_ID', 'C'])
        self.con_trial = self.con_trial.merge(
            Con_table,
            on='Con_ID',
            how='left'
        )
        self.k = k
        con_summary = self.con_trial.groupby(['Con_ID', 'Stim', 'Chan', 'C'], as_index=False)[['Sig', 'd', 'LL']].mean()
        con_summary = self.check_sleep_cluster(con_summary)
        con_summary.to_csv(os.path.join(path_output, 'con_summary.csv'), header=True, index=False)
        self.con_summary = con_summary

    # Format x-axis tick labels as daytime hours
    def check_sleep_cluster(self, con_summary):
        from scipy.stats import mannwhitneyu
        from scipy.stats import ranksums
        self.con_trial = bf.add_sleepstate(self.con_trial)
        # todo: z-score by Wake data

        for c in np.unique(self.con_trial.C):
            for sleep in ['NREM', 'REM']:
                val_wake = self.con_trial.loc[
                    (self.con_trial.SleepState == 'Wake') & (self.con_trial.C == c), 'LL_sig'].values
                val_sleep = self.con_trial.loc[
                    (self.con_trial.SleepState == sleep) & (self.con_trial.C == c), 'LL_sig'].values
                t, p = ranksums(val_wake,
                                val_sleep)  # scipy.stats.ttest_ind(val_wake,val_sleep) #U1, p = mannwhitneyu(val_wake, val_sleep)
                if p < 0.01:
                    con_summary.loc[con_summary.C == c, sleep] = -np.sign(t)
                else:
                    con_summary.loc[con_summary.C == c, sleep] = 0
        return con_summary

    def format_time_hour(self, x, pos):
        while x > 24:
            x -= 24
        return f'{int(x):02d}:00'

    def calculate_cophenetic_corr(self, A):
        """
        Compute the cophenetic correlation coefficient for matrix A.

        Parameters:
        - A : numpy.ndarray
            Input matrix.

        Returns:
        - float
            Cophenetic correlation coefficient.

            The cophenetic correlation coefficient is measure which indicates the dispersion of the consensus matrix and is based on the average of connectivity matrices.
            It measures the stability of the clusters obtained from NMF. It is computed as the Pearson correlation of two distance matrices:
            the first is the distance between samples induced by the consensus matrix; the second is the distance between samples induced by the linkage used in the reordering of the consensus matrix [Brunet2004].

        """

        # Extract the values from the lower triangle of A
        avec = np.array([A[i, j] for i in range(A.shape[0] - 1)
                         for j in range(i + 1, A.shape[1])])

        # Consensus entries are similarities, conversion to distances
        # 1. matrix: distance between samples indced by consensus matrix
        Y = 1 - avec

        # Hierarchical clustering
        # 2. matrix: distance between samples induced by the linkage used in the reordering of the consensus matrix
        Z = linkage(Y, method='average')

        # Cophenetic correlation coefficient of a hierarchical clustering
        coph = cophenet(Z, Y)[0]

        return coph

    def plot_pearson_hypnogram(self, hyp_style='Block'):
        from matplotlib.gridspec import GridSpec
        path_file = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'figures')
        M, label = BM_func.cal_correlation_condition(self.con_trial, metric='LL', condition='Block')
        if hyp_style == 'Block':
            hypnogram = np.zeros((len(label),))
            for ix, l in enumerate(label):
                hypnogram[ix] = np.bincount(self.con_trial.loc[self.con_trial.Block == l, 'Sleep']).argmax()
        else:
            stimlist_hypno = pd.read_csv(os.path.join(self.path_patient_analysis, '/stimlist_hypnogram.csv'))
        x_ax = np.arange(len(label))

        # Create figure and subplots using gridspec
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 2, width_ratios=[5, 0.5], height_ratios=[1, 5], wspace=0.2, hspace=0.2)

        # Subplot for the correlation matrix
        ax = plt.subplot(gs[1, 0])
        # ax.set_aspect('equal')
        ax_h = plt.subplot(gs[0, 0], sharex=ax)
        ax_cbar = plt.subplot(gs[1, 1])

        ax_h.plot(x_ax, hypnogram, c='black', linewidth=2)
        ax_h.axhspan(-1, 0.2, color=color_elab[0, :])  # Using color map for color
        ax_h.fill_between(x_ax, hypnogram, -1, color=color_elab[0, :])  # Using color map for color
        ax_h.set_yticks([0, 1, 2, 3, 4])
        ax_h.set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
        ax_h.set_ylim([-1, 5])
        ax_h.invert_yaxis()
        ax_h.set_xticks([])
        ax_h.set_ylabel('Score', fontsize=10)
        ax_h.set_xlim(x_ax[0], x_ax[-1])

        # Plot Pearson correlation matrix (M)
        im = ax.pcolormesh(M, cmap='jet', vmin=np.percentile(M, 10), vmax=np.percentile(M, 90))
        ax.set_ylabel('Block Number', fontsize=10)
        ax.set_xlabel('Block Number', fontsize=10)
        ax.set_xlim(x_ax[0], x_ax[-1])

        # Add a colorbar to the last subplot
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.ax.set_ylabel('Pearsons Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(path_file, 'Block_pearson.jpg'), dpi=300)
        plt.show()

    def plot_basis(self):
        path_file = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'figures')
        matrix_data = self.W

        # con_summary = self.con_trial.groupby(['Con_ID'], as_index=False)['Sig'].mean()
        # non_zero_rows = self.con_summary.loc[self.con_summary.Sig > self.sig_thr, 'Con_ID']
        # matrix_data = matrix_data[non_zero_rows]
        # Normalize the filtered matrix row-wise and column-wise
        row_min = matrix_data.min(axis=1, keepdims=True)
        row_max = matrix_data.max(axis=1, keepdims=True)
        row_normalized = (matrix_data - row_min) / (row_max - row_min)

        col_min = matrix_data.min(axis=0, keepdims=True)
        col_max = matrix_data.max(axis=0, keepdims=True)
        col_normalized = (matrix_data - col_min) / (col_max - col_min)
        # remove zero
        # Create a figure with 1 row and 3 columns for subplots
        n_con = matrix_data.shape[0]
        n_cluster = matrix_data.shape[1]
        fig, axs = plt.subplots(1, 3, figsize=(15, 10))
        # Plot the original matrix with a colorbar
        im1 = axs[0].pcolormesh(matrix_data, cmap='hot', vmin=np.percentile(matrix_data, 10),
                                vmax=np.percentile(matrix_data, 90))
        axs[0].set_title('Original Matrix')
        fig.colorbar(im1, ax=axs[0])

        # Plot the row-wise normalized matrix with a colorbar
        im2 = axs[1].pcolormesh(row_normalized, cmap='viridis', vmin=0, vmax=1)
        axs[1].set_title('Row-wise Normalized')
        fig.colorbar(im2, ax=axs[1])

        # Plot the column-wise normalized matrix with a colorbar
        im3 = axs[2].pcolormesh(col_normalized, cmap='viridis', vmin=0, vmax=1)
        axs[2].set_title('Column-wise Normalized')
        fig.colorbar(im3, ax=axs[2])

        # Adjust spacing between subplots
        plt.tight_layout()
        plt.savefig(os.path.join(path_file, 'Basis_function.jpg'), dpi=300)
        plt.show()

    def plot_consensus_best(self):
        from matplotlib.gridspec import GridSpec
        path_file = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'figures')
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'conNMF')

        path = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'data')
        con_trial = self.con_trial

        Mp, _ = BM_func.cal_correlation_condition(con_trial, metric='LL', condition='Block')
        label = np.unique(con_trial.Block)
        M = pd.read_csv(os.path.join(path_output, 'k=' + str(self.k), 'consensus_matrix.csv'), header=None).values
        hypnogram = np.zeros((len(label),))

        for ix, l in enumerate(label):
            hypnogram[ix] = np.bincount(con_trial.loc[con_trial.Block == l, 'Sleep']).argmax()
        x_ax = np.arange(len(label))

        coph = self.calculate_cophenetic_corr(M)
        p = np.corrcoef(M.flatten(), Mp.flatten())[0, 1]
        # Create figure and subplots using gridspec
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle(self.subj + ' -- Consensus Matrix Rank ' + str(self.k) + ', coph: ' + str(
            np.round(coph, 2)) + ', P(corr): ' + str(np.round(p, 2)))
        gs = GridSpec(2, 2, width_ratios=[5, 0.5], height_ratios=[1, 5], wspace=0.2, hspace=0.2)

        # Subplot for the correlation matrix
        ax = plt.subplot(gs[1, 0])
        # ax.set_aspect('equal')
        ax_h = plt.subplot(gs[0, 0], sharex=ax)
        ax_cbar = plt.subplot(gs[1, 1])

        ax_h.plot(x_ax, hypnogram, c='black', linewidth=2)
        ax_h.axhspan(-1, 0.2, color=color_elab[0, :])  # Using color map for color
        ax_h.fill_between(x_ax, hypnogram, -1, color=color_elab[0, :])  # Using color map for color
        ax_h.set_yticks([0, 1, 2, 3, 4])
        ax_h.set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
        ax_h.set_ylim([-1, 5])
        ax_h.invert_yaxis()
        ax_h.set_xticks([])
        ax_h.set_ylabel('Score', fontsize=10)
        ax_h.set_xlim(x_ax[0], x_ax[-1])

        # Plot Pearson correlation matrix (M)
        im = ax.pcolormesh(M, cmap='pink', vmin=np.percentile(M, 10), vmax=np.percentile(M, 90))
        ax.set_ylabel('Block Number', fontsize=10)
        ax.set_xlabel('Block Number', fontsize=10)
        ax.set_xlim(x_ax[0], x_ax[-1])

        # Add a colorbar to the last subplot
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.ax.set_ylabel('Probability of shared membership (100 runs)')
        plt.tight_layout()
        plt.savefig(os.path.join(path_file, 'Block_consensus_hypnogram.jpg'), dpi=300)
        plt.show()

    def plot_consensus_examples(self, ranks=[4, 6, 8, 10]):
        path_file = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'figures')
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'conNMF')
        n_plots = len(ranks)

        # Determine the number of rows and columns for subplots
        n_rows = int(np.sqrt(n_plots))
        n_cols = int(np.ceil(n_plots / n_rows))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
        plt.suptitle(self.subj + ' -- NMF Consensus Matrix')
        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        label = 'Block Number'

        for k, ax in zip(ranks, axs.flatten()):
            M_cons = pd.read_csv(os.path.join(path_output, 'k=' + str(k), 'consensus_matrix.csv'), header=None).values
            coph = self.calculate_cophenetic_corr(M_cons)

            im = ax.pcolormesh(M_cons, cmap='pink', vmin=0.1, vmax=0.9)
            ax.set_title('Rank ' + str(k) + ', coph: ' + str(np.round(coph, 2)))

            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel(label)
            else:
                ax.set_xticks([])
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(label)
            else:
                ax.set_yticks([])

        # Add a colorbar to the last subplot
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel('Probability of shared membership (100 runs)')
        plt.savefig(os.path.join(path_file, 'Consensus.jpg'), dpi=300)
        plt.show()
        print('stop')

    def plot_cluster_BM(self, file_end=''):
        import matplotlib
        cmap = matplotlib.colormaps["Accent"]
        cmap.set_bad(color='black')

    def plot_BM_clusters(self, assign_method='max', file_end=''):
        import matplotlib

        if assign_method == 'max':
            Wmax = np.argmax(self.W, 1)
        else:
            Wmax = NMFf.assgin_cluster(self.V, self.H)  # np.argmax(W,1)
        BM_W = np.zeros((len(self.labels_all), len(self.labels_all))) - 1
        for w_ix in range(self.H.shape[0]):
            for con_ix, con in enumerate(np.unique(self.con_summary.Con_ID)):
                if Wmax[con_ix] == w_ix:
                    #  coeff = W[con_ix, w_ix]
                    sc = self.con_summary.loc[self.con_summary.Con_ID == con, 'Stim'].values[0].astype('int')
                    rc = self.con_summary.loc[self.con_summary.Con_ID == con, 'Chan'].values[0].astype('int')
                    BM_W[sc, rc] = w_ix
        BM_W = BM_W.astype('int')
        cmap = plt.get_cmap('Accent', self.H.shape[0])  # matplotlib.colormaps["Accent"]
        cmap.set_bad(color='black')
        # mask some 'bad' data, in your case you would have: data == 0
        BM_W = np.ma.masked_where(BM_W == -1, BM_W)

        fig = plt.figure(figsize=(10, 10))
        plt.suptitle('Clusters - ' + assign_method)
        axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        axcolor = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        BM_plots.plot_BM(BM_W, self.labels_all, self.hemisphere, axmatrix, axcolor=axcolor, cmap=cmap,
                         vlim=[np.min(BM_W) - 0.5, np.max(BM_W) + 0.5], sort=1, cat=1)
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'figures')
        plt.savefig(os.path.join(path_output, 'Cluster_BM_' + assign_method + file_end + '.svg'))
        plt.show()
        print('stop')

    def plot_H_hypnogram(self, file_end=''):
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
        cmap = plt.get_cmap('Accent', rk)  # matplotlib.colormaps["Accent"]
        cmap.set_bad(color='black')
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
            ax.plot(x_ax_block, self.H[i], linewidth=4, label=f'H{i + 1}', color=cmap(i))
        # ax.set_xticks(x_ticks_h)
        # ax.set_xticklabels(labels_hour, fontsize=12)
        ax.set_ylabel('H Coefficients', fontsize=15)
        ax.set_xlabel('Time', fontsize=15)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_time_hour))
        # same xlim as ax_h to be perfectly aligned
        ax.set_xlim(ax_h.get_xlim())

        plt.tight_layout()
        plt.savefig(os.path.join(path_output, 'H_hypnogram' + file_end + '.svg'))

    def plot_clusters_connectogram(self, file_end=''):
        cwp = os.getcwd()

        plotter = ConnectogramPlotter()
        path_output = os.path.join(self.path_patient_analysis, 'BrainMapping', 'CR', 'NMF', 'cluster_connectogram')
        os.makedirs(path_output, exist_ok=True)  # Create directories if they don't exist

        con_table = self.con_trial.groupby(['Con_ID', 'Chan', 'Stim', 'C'], as_index=False)[['LL', 'Sig', 'd']].mean()
        con_table.Chan = con_table.Chan.astype('int')
        con_table.Stim = con_table.Stim.astype('int')
        con_table.insert(0, 'Subj', self.subj)
        # Create new columns 'StimA' and 'ChanA' using vectorized operations
        con_table['StimA'] = [re.sub(r'\d', '', label) for label in self.labels_all[con_table['Stim']]]
        con_table['ChanA'] = [re.sub(r'\d', '', label) for label in self.labels_all[con_table['Chan']]]

        os.chdir(path_connectogram)

        n_cluster = self.H.shape[0]
        for ix_cluster, cluster in enumerate(range(n_cluster)):  #
            # cluster_chans = self.column_cluster_assignments[cluster]
            # con_sel = con_table.loc[np.isin(con_table.Con_ID, cluster_chans)]
            con_sel = con_table.loc[(con_table.C == ix_cluster)].reset_index(drop=True)
            path_save = os.path.join(path_output, 'Cluster_' + str(cluster + 1) + file_end + '.jpg')
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
    # run_main.plot_basis()
    #run_main.plot_consensus_best()
    run_main.plot_pearson_hypnogram(hyp_style = 'full')
    ## run_main.plot_consensus_examples()
    #run_main.plot_BM_clusters('_conNMF')
    #run_main.plot_H_hypnogram('_conNMF')
    # run_main.plot_clusters_connectogram('_conNMF')
    # run_main.plot_LL_clusters()

    print(subj + ' ----- DONE')


thread = 0
subjs = ['EL011', 'EL012', 'EL013', 'EL015', 'EL014', 'EL016', 'EL017', "EL019", "EL020", "EL021", "EL022", "EL025",
         "EL026"]

subjs = ["EL011", "EL014", "EL015", "EL016", "EL017", "EL019", "EL020",
         "EL021", "EL022", "EL025", "EL027"]
subjs = ["EL027"]
for subj in subjs:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'"EL021", "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016",
    if thread:
        _thread.start_new_thread(start_subj, (subjs))
    else:
        start_subj(subj)
if thread:
    while 1:
        time.sleep(1)

print('Done')
