import os
import numpy as np
import mne
import h5py
# import scipy.fftpack
import matplotlib
import pywt
import tqdm
from matplotlib.ticker import ScalarFormatter
import platform
import matplotlib.pyplot as plt
# from scipy import signal
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
import LL_funcs
import freq_funcs
import plot_WT_funcs
import re
import load_summary as ls
from matplotlib.gridspec import GridSpec
import mne
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from math import sin
import dabest
import LL_funcs
from pandas import read_excel
from scipy.stats import norm

from scipy.io import savemat
import scipy.io
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

dist_groups = np.array([[0, 15], [15, 30], [30, 5000]])
dist_labels = ['local (<15 mm)', 'short (<30mm)', 'long']


color_d, color_dist, color_group, color_elab = ls.get_color()

def scatter_prob(data_plot, c= 'd',  m = 'DI'):
    if c =='d':
        colormap_d = ListedColormap(color_d)
    else:
        colormap_d = 'hot'
    # data_plot = data_DI.loc[(~np.isnan(data_DI.P_BA)) & (~np.isnan(data_DI.P_AB)) & (~np.isnan(data_DI[m]))]
    norm = plt.Normalize(data_plot[c].min(), 100)
    # sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    data_plot.loc[data_plot.P_AB == 0, 'P_AB'] = np.random.random(len(data_plot.loc[data_plot.P_AB == 0]))/20 - 0.05
    data_plot.loc[data_plot.P_BA == 0, 'P_BA'] = np.random.random(len(data_plot.loc[data_plot.P_BA == 0])) / 20 - 0.05
    fig = plt.figure(figsize=(14, 10))
    g = sns.scatterplot(x='P_AB', y='P_BA', c=data_plot[c], s=50, data=data_plot, cmap=colormap_d)
    scalarmappaple = cm.ScalarMappable(norm=norm, cmap=colormap_d)
    fig.colorbar(scalarmappaple)
    plt.title('Directionality Index (' + m + ')', fontsize=25)
    plt.xticks([0, 0.5, 1], ['0', '50', '100'], fontsize=20)
    plt.yticks([0, 0.5, 1], ['0', '50', '100'], fontsize=20)
    plt.xlabel('Probability AB [%]', fontsize=25)
    plt.ylabel('Probability BA [%]', fontsize=25)

    plt.savefig(
        sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\P_scatter_'+c+'.svg')
    plt.savefig(
        sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\P_scatter_'+c+'.jpg')
    plt.show()

    # adding kde
    grid = sns.JointGrid(x='P_AB', y='P_BA', data=data_plot, height=15, ratio=4)
    g = grid.plot_joint(sns.scatterplot, c=data_plot[c], s=50, data=data_plot, cmap=colormap_d)
    # plt.pcolor(np.linspace(0, 1, 101), np.linspace(0, 1, 101),abs(M), cmap='binary', vmin=0.1, vmax=0.8, alpha=1)
    g.ax_joint.set_xticks([0, 0.5, 1])
    g.ax_joint.set_xticklabels([0, 50, 100], fontsize=20)
    g.ax_joint.set_yticks([0, 0.5, 1])
    g.ax_joint.set_yticklabels([0, 50, 100], fontsize=20)
    g.ax_joint.set_ylabel('Probability BA [%]', fontsize=20)
    g.ax_joint.set_xlabel('Probability AB [%]', fontsize=20)

    sns.kdeplot(x='P_AB', data=data_plot[data_plot.P_BA <= 0], hue='Dist', hue_order=dist_labels, ax=g.ax_marg_x,
                legend=False, linewidth=10, palette=color_dist)
    sns.kdeplot(x='P_BA', data=data_plot[data_plot.P_AB <= 0], hue='Dist', hue_order=dist_labels, ax=g.ax_marg_y,
                vertical=True, legend=True, linewidth=10, palette=color_dist)
    plt.savefig(
        sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\P_scatter_' + c + '_kde.svg')
    plt.show()

def hist_groups(data_plot, color_group, group_labels, m='DI', group = 'Dist'):
    y_cut_lim = np.array([0, 500, np.ceil(np.max(data_plot.groupby(group)[m].count())/500)*500])
    # group_labels = np.unique(data_plot[group])
    for dist_sel, i in zip(group_labels, np.arange(len(group_labels))):
        f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace': 0.05})
        data_plotc = data_plot[data_plot[group] == dist_sel]
        data_plotc = data_plotc.reset_index(drop=True)
        sns.histplot(data=data_plotc, x=abs(data_plotc[m]),
                     binwidth=0.1, ax=ax_top, color=color_group[i])
        sns.histplot(data=data_plotc, x=abs(data_plotc[m]),
                     binwidth=0.1, ax=ax_bottom, color=color_group[i])
        ax_top.set_ylim(y_cut_lim[1], y_cut_lim[2])  # those limits are fake
        ax_bottom.set_ylim(y_cut_lim[0], y_cut_lim[1])
        sns.despine(ax=ax_bottom)
        sns.despine(ax=ax_top, bottom=True)
        plt.title(dist_sel)
        l = "".join(re.findall("[a-z]+", dist_sel))
        plt.savefig(
            sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\\'+m+'_hist_' + l + '.svg')
        plt.savefig(
            sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\\'+m+'_hist_' + l + '.jpg')
        plt.show()

def violin_groups(data_plot, xx, yy, cc, palette, labels):
    sns.set_style('white')
    f, ax = plt.subplots(figsize=(15, 10))
    ax = sns.violinplot(x=xx, y=yy, data=data_plot, dodge=False,
                        palette=palette,
                        scale="width", inner=None, alpha=.2)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
        violin.set_alpha(0.8)

    sns.boxplot(x=xx, y=yy, data=data_plot, saturation=1, showfliers=False,
                width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax, linewidth=3)
    old_len_collections = len(ax.collections)
    if cc:
        sns.stripplot(x=xx, y=yy, c = data_plot[cc].values, data=data_plot, dodge=False, ax=ax, alpha=0.1, cmap='hot')
    else:
        sns.stripplot(x=xx, y=yy, data=data_plot, palette=palette, dodge=False, ax=ax, alpha=0.1)
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(labels[0], fontsize=24)
    plt.xlabel("", fontsize=24)
    # ax.legend_.remove()
    plt.savefig(
        sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\\violin_' + xx + '_' + yy + '.svg')
    plt.savefig(
        sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\DI\\violin_' + xx + '_' + yy + '.jpg')
    plt.show()

def violin_pearson(data_plot, weight= 'LL', xx='SleepState', yy='P'):
    sns.set_style('white')
    f, ax = plt.subplots(figsize=(15, 10))
    ax = sns.violinplot(x=xx, y=yy, data=data_plot, dodge=False,
                        scale="width", inner=None, alpha=.02)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
        violin.set_alpha(0.5)

    sns.boxplot(x=xx, y=yy, data=data_plot, saturation=1, showfliers=False,
                width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax, linewidth=3)
    old_len_collections = len(ax.collections)

    sns.swarmplot(x=xx, y=yy, data=data_plot, hue='Subj', dodge=False, ax=ax, alpha=0.8, s=10)

    #sns.stripplot(x=xx, y=yy, data=data_plot, dodge=False, ax=ax, alpha=0.8, s=10)
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Connectivity Weight: '+weight, fontsize=24)
    plt.ylabel('Pearson Correlation to Wake', fontsize=24)
    plt.xlabel("", fontsize=24)
    # ax.legend_.remove()
    plt.savefig(
        sub_path + '\EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\Sleep\degree\\Pearson_network_' + weight + '.svg')
    plt.savefig(
        sub_path + '\EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\Sleep\degree\\Pearson_network_' + weight + '.jpg')
    plt.show()