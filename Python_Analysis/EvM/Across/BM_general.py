import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd

import sys

sys.path.append('T:\EL_experiment\Codes\CCEP_human\Python_Analysis/py_functions')

from scipy.stats import norm
from tkinter import *
import scipy
from scipy import signal
import matplotlib.cm as cm

import platform
from glob import glob
from scipy.spatial import distance
import basic_func as bf
from scipy.integrate import simps
from numpy import trapz
import IO_func as IOF
import BM_func as BMf
import tqdm
from matplotlib.patches import Rectangle
from pathlib import Path
import LL_funcs as LLf
import freq_funcs as ff
#
from scipy.signal import hilbert, butter, filtfilt
import scipy.stats as stats
from tqdm.notebook import trange, tqdm
import significance_funcs as sig_func
import load_summary as ls
import BM_across_plots as plotting

cwd = os.getcwd()

##all
cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]
dist_groups = np.array([[0, 15], [15, 30], [30, 5000]])
dist_labels = ['local (<15 mm)', 'short (<30mm)', 'long']
group_labels = ['local direct', 'long direct', 'indirect']
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

##
subjs = ['EL010', 'EL011', 'EL012', 'EL013', 'EL014', 'EL015', 'EL016', 'EL017', 'EL019', 'EL020', 'EL021']

DI_file = sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\M_dir_all.csv'
con_file = sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\data_con.csv'

###load data
if not os.path.exists(con_file):
    ls.get_connections(subjs, sub_path, con_file)

if not os.path.exists(DI_file):
    ls.get_DI(subjs, sub_path, DI_file)

data_DI = pd.read_csv(DI_file)
data_con = pd.read_csv(con_file)
if not 'onset_B' in data_DI:
    data_DI2 = ls.update_DI(data_con)
    data_DI = data_DI.merge(data_DI2, on=['Subj', 'A', 'B', 'd', 'P_AB', 'P_BA', 'DI'], how='outer')
    data_DI.to_csv(DI_file, header=True, index=False)
print('Data loaded')

## plots

# get color
color_d, color_dist, color_group, color_elab = ls.get_color()

data_con.insert(6, 'Group', 'indirect')
data_con.loc[(data_con.d < 20) & (data_con.onset < 0.02), 'Group'] = 'local direct'
data_con.loc[(data_con.d > 20) & (data_con.onset < 0.02), 'Group'] = 'long direct'
# data_plot.insert(3, 'Dist', 'local (<15 mm)')
# data_plot = data_plot.reset_index(drop=True)
# data_plot.loc[data_plot.d > 15, 'Dist'] = 'short (<30mm)'
# data_plot.loc[data_plot.d > 30, 'Dist'] = 'long'

##
# todo: create function in another file
plot = 1
if plot:
    m = 'DI'
    data_plot = data_DI.loc[(~np.isnan(data_DI.P_BA)) & (~np.isnan(data_DI.P_AB)) & (~np.isnan(data_DI[m]))]
    ### DI by distance
    data_plot.insert(3, 'Dist', 'local (<15 mm)')
    data_plot = data_plot.reset_index(drop=True)
    data_plot.loc[data_plot.d > 15, 'Dist'] = 'short (<30mm)'
    data_plot.loc[data_plot.d > 30, 'Dist'] = 'long'


    # plotting.scatter_prob(data_plot, c='d', m=m)



    # plotting.hist_groups(data_plot, color_dist, dist_labels,  m='DI', group='Dist')

    # data_plot = data_con.loc[(data_con.Sig>0)]
    # data_plot = data_plot.reset_index(drop=True)
    # group_labels = ['local direct', 'long direct', 'indirect']
    # plotting.hist_groups(data_plot, color_group, group_labels, m='DI', group='Group')

    ### vilin plot in groups

    plot = 1
    if plot:
        for xx, col in zip(['Group', 'Dist'],[color_group, color_dist]):
            # yy = 'DI'
            for yy, cc in zip(['DI', 'Sig'], ['LL_sig', 'LL_sig']):
                ylabel = "Probability"
                if yy == 'DI':
                    ylabel = 'Directionality Index'
                data_plot = data_con  # [data_con_all.Sig>0]
                if yy == 'Sig':
                    data_plot = data_plot[data_plot.Sig > 0]
                data_plot = data_plot.reset_index(drop=True)
                data_plot[yy] = abs(data_plot[yy])

                plotting.violin_groups(data_plot, xx, yy, None, col, ylabel)

        # fig = plt.figure(figsize=(15, 15))
        # ax = fig.add_subplot(111, projection='3d')
        # data_plot = data_con[data_con.Sig > 0]
        # data_plot = data_plot.reset_index(drop=True)
        # X, Y, Z = data_plot.onset.values, data_plot.d.values, data_plot.Sig.values
        # scatter_plot = ax.scatter(X, Y, np.array([Z]), c=Z, cmap='copper', alpha=0.5)
        # plt.colorbar(scatter_plot)
        # plt.xlabel('onset [s]', fontsize=20)
        # plt.ylabel('distance [mm]', fontsize=20)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.savefig(sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\P_3d.svg')
        # plt.savefig(sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\P_3d.jpg')
        # plt.show()

    ##
    # data_DI3 = data_DI[
    #     (data_DI.d > 20) & (~np.isnan(data_DI.DI)) & (~np.isnan(data_DI.onset_A)) & (~np.isnan(data_DI.onset_B))]
    # data_DI3 = data_DI3.reset_index(drop=True)
    # xl = 0.03
    # fig, axs = plt.subplots(ncols=4, nrows=2, sharex=False, sharey=False,
    #                         gridspec_kw={'width_ratios': [1, 3, 0.2, 0.8], 'height_ratios': [3, 1]}, figsize=(12, 10))
    # for ax in fig.get_axes():
    #     ax.label_outer()
    #
    # for i in range(2):
    #     for j in range(4):
    #         axs[i, j].set_xlabel('')
    #         axs[i, j].set_ylabel('')
    #         axs[i, j].set_facecolor('xkcd:black')
    #         axs[i, j].set_xticks([])
    #         axs[i, j].set_yticks([])
    #
    # ax = sns.scatterplot(x=data_DI3.onset_A, y=data_DI3.onset_B, c=data_DI3.DI.values, ax=axs[1, 0], cmap='seismic')
    #
    # ax.set_xlim([-0.001, xl])
    # ax.set_ylim([-0.001, xl])
    # axs[1, 0].set_yticks(np.arange(0, xl + 0.001, 0.005))
    # axs[1, 0].set_xticks(np.arange(0, xl + 0.001, 0.005))
    #
    # ax = sns.scatterplot(x=data_DI3.onset_A, y=data_DI3.onset_B, c=data_DI3.DI.values, ax=axs[1, 1], cmap='seismic')
    #
    # ax.set_xlim([xl, 0.3])
    # ax.set_ylim([-0.001, xl])
    # axs[1, 1].set_xticks(np.arange(0.05, 0.31, 0.05))
    # axs[1, 1].set_yticks([])
    #
    # ax = sns.scatterplot(x=data_DI3.onset_A, y=data_DI3.onset_B, c=data_DI3.DI.values, ax=axs[0, 1], cmap='seismic')
    #
    # ax.set_xlim([xl, 0.3])
    # ax.set_ylim([xl, 0.3])
    # axs[0, 1].set_yticks([])
    # axs[0, 1].set_xticks([])
    #
    # ax = sns.scatterplot(x=data_DI3.onset_A, y=data_DI3.onset_B, c=data_DI3.DI.values, ax=axs[0, 0], cmap='seismic')
    #
    # ax.set_xlim([-0.001, xl])
    # ax.set_ylim([xl, 0.3])
    # axs[0, 0].set_yticks(np.arange(0.05, 0.31, 0.05))
    #
    # data_DI3 = data_DI[(data_DI.onset_A >= 0) & (data_DI.d > 20) & (data_DI.DI == 1)]
    # data_DI3 = data_DI3.reset_index(drop=True)
    # data_DI3.onset_B = 0.4 + np.random.random(len(data_DI3)) / 10
    #
    # ax = sns.scatterplot(x=data_DI3.onset_B, y=data_DI3.onset_A, c=data_DI3.DI.values, ax=axs[0, 3], cmap='seismic')
    #
    # ax.set_xlim([0.39, 0.51])
    # ax.set_ylim([xl, 0.3])
    # axs[0, 3].set_yticks([])
    # axs[1, 3].set_xticks([])
    #
    # ax = sns.scatterplot(x=data_DI3.onset_B, y=data_DI3.onset_A, c=data_DI3.DI.values, ax=axs[1, 3], cmap='seismic')
    #
    # axs[1, 3].set_xlim([0.39, 0.51])
    # axs[1, 3].set_ylim([xl, 0.3])
    # axs[1, 3].set_xlabel('Unidirectional')
    #
    # plt.subplots_adjust(wspace=0.0001, hspace=0.001)
    # plt.savefig(sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\Cat_onset.svg')
    # plt.savefig(sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\Cat_onset.jpg')
    # plt.show()

####
#
# data_DI3 = data_DI[
#     (data_DI.d > 20) & (~np.isnan(data_DI.DI)) & (~np.isnan(data_DI.onset_A)) & (~np.isnan(data_DI.onset_B))]
# data_DI3 = data_DI3.reset_index(drop=True)
# data_DI3.DI = abs(data_DI3.DI)
# data_DI3.insert(4, 'Cat', 'D-I')
# t = 0.03
# data_DI3.loc[(data_DI3.onset_B > t) & (data_DI3.onset_A > t), 'Cat'] = 'I-I'
# data_DI3.loc[(data_DI3.onset_B < t) & (data_DI3.onset_A < t), 'Cat'] = 'D-D'
# data_DI3.loc[(np.isnan(data_DI3.onset_A)), 'Cat'] = 'U'
# data_DI3.loc[(np.isnan(data_DI3.onset_B)), 'Cat'] = 'U'
# xx = 'Cat'
# yy = 'DI'
# ylabel = "Probability"
# if yy == 'DI':
#     ylabel = 'Directionality Index'
# data_plot = data_DI3  # [data_con_all.Sig>0]
# data_plot[yy] = abs(data_plot[yy])
#
# sns.set_style('white')
# palette = color_dist
# fig, ax = plt.subplots(figsize=(15, 10))
# ax = sns.violinplot(x=xx, y=yy, data=data_plot, dodge=False,
#                     palette=palette,
#                     scale="width", inner=None, alpha=.2)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     bbox = violin.get_paths()[0].get_extents()
#     x0, y0, width, height = bbox.bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
#     violin.set_alpha(0.8)
#
# sns.boxplot(x=xx, y=yy, data=data_plot, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax, linewidth=3)
# old_len_collections = len(ax.collections)
# sns.stripplot(x=xx, y=yy, data=data_plot, palette=palette, dodge=False, ax=ax, alpha=0.1)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.ylabel(ylabel, fontsize=24)
# plt.xlabel("", fontsize=24)
# plt.savefig(sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\DI_violin_Cat.svg')
# plt.savefig(sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\CI_violin_Cat.jpg')
# plt.show()
