import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import seaborn as sns
import copy
import pandas as pd
import platform
from matplotlib.colors import ListedColormap
import re

# regions = pd.read_excel("T:\EL_experiment\Patients\\all\elab_labels.xlsx", sheet_name='regions', header=0)
#
# color_regions = regions.color.values
# regions_G = regions.subregion.values
# regions = regions.label.values

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]

color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])

# fig = plt.figure(figsize=(25, 25))
# fig.patch.set_facecolor('xkcd:white')
# axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])  # x, y, (start posiion), lenx, leny
CIRC_AREAS_FILEPATH = "X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\\circ_areas.xlsx"
color_regions = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='plot')
atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')

plt.rcParams.update({
    'font.family': 'arial',
    'font.size': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'svg.fonttype': 'none',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 10
})


def clean_label(labels):
    """Remove spaces and numbers from labels."""
    labels = [label.replace(" ", "") for label in labels]
    labels = [re.sub(r'\d', '', label) for label in labels]
    return labels


def remove_bad_areas(M, labels, regions, hemisphere, axis=-1):
    """Filter out bad regions from the matrix and labels."""
    BAD_REGIONS = {'WM', 'OUT', 'out', 'Putamen', 'Pallidum', 'Necrosis', 'LV', 'U', 'Unknown'}

    valid_indices = [i for i, region in enumerate(regions) if region not in BAD_REGIONS]
    if axis == -1:
        M = M[valid_indices][:, valid_indices]
    elif axis == 0:
        M = M[valid_indices, :]
    else:
        M = M[:, valid_indices]
    labels = [labels[i] for i in valid_indices]
    regions = [regions[i] for i in valid_indices]
    hemisphere = [hemisphere[i] for i in valid_indices]
    return M, labels, regions, hemisphere


def get_region(labels, atlas):
    """Retrieve the region for each label from the atlas."""
    return [atlas.loc[atlas.Abbreviation == l, 'Region'].values[0] if l in atlas.Abbreviation.values else 'Unknown' for
            l in labels]


def sort_M(M, labels, regions, hemisphere, axis=-1):
    areas_sel_sort = [h + '_' + r for h, r in zip(hemisphere, regions)]
    ind = np.argsort(areas_sel_sort)

    if axis == -1:
        M = M[ind][:, ind]
    elif axis == 0:
        M = M[ind]
    else:
        M = M[:, ind]
    labels = [labels[i] for i in ind]
    regions = [regions[i] for i in ind]
    hemisphere = [hemisphere[i] for i in ind]

    return M, labels, regions


def plot_BM(M, labels, hemisphere, axmatrix, axcolor=None, cmap='hot', vlim=None, sort=1, cat=0):
    """Plot the connectivity map."""
    labels = clean_label(labels)
    areas = get_region(labels, atlas)
    M, labels, areas, hemisphere = remove_bad_areas(M, labels, areas, hemisphere)

    if vlim is None:
        vlim = [np.nanpercentile(M, 5), np.nanpercentile(M, 95)]

    if sort:
        M, labels, areas = sort_M(M, labels, areas, hemisphere)
    labels = [h + '_' + r for h, r in zip(hemisphere, labels)]
    n_nodes = M.shape[0]

    im = axmatrix.pcolormesh(M, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    # im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    axmatrix.set_xlim([-1.5, n_nodes - 0.5])
    axmatrix.set_ylim([-0.5, n_nodes + 0.5])
    axmatrix.set_xticks(np.arange(n_nodes) + 0.5)  # Adjust to center of the grid cell
    axmatrix.set_xticklabels(labels, rotation=90, fontsize=8)
    axmatrix.set_yticks(np.arange(n_nodes) + 0.5)  # Adjust to center of the grid cell
    axmatrix.set_yticklabels(labels, fontsize=8)
    # Add colored rectangles for regions
    for i in range(n_nodes):
        r = areas[i]
        color = color_regions.loc[color_regions.Area == r, 'color'].values[0]
        axmatrix.add_patch(Rectangle((i - 0.5, n_nodes - 0.5), 1, 1, alpha=1, facecolor=color))
        axmatrix.add_patch(Rectangle((-1.5, i - 0.5), 1, 1, alpha=1, facecolor=color))

    if axcolor:
        if cat:
            plt.colorbar(im, cax=axcolor, ticks=np.arange(vlim[0] + 0.5, vlim[1] + 0.5))
        else:
            plt.colorbar(im, cax=axcolor)


def plot_BM_coeff(M, labels, hemisphere, axmatrix, axcolor=None, cmap='hot', vlim=None, sort=1, orientation=0):
    """Plot the connection x coeffcient. E.g. NMF W basis function"""

    labels = clean_label(labels)
    areas = get_region(labels, atlas)
    M, labels, areas, hemisphere = remove_bad_areas(M, labels, areas, hemisphere, orientation)

    if vlim is None:
        vlim = [np.nanpercentile(M, 5), np.nanpercentile(M, 95)]

    if sort:
        M, labels, areas = sort_M(M, labels, areas, hemisphere, orientation)
    labels = [h + '_' + r for h, r in zip(hemisphere, labels)]
    if orientation == 0:
        n_nodes = M.shape[0]
    else:
        n_nodes = M.shape[1]

    im = axmatrix.pcolormesh(M, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    # im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    if orientation == 0:  # if nodes are in rows, show labels in y-axis (left)
        axmatrix.set_ylim([-0.5, n_nodes + 0.5])
        axmatrix.set_yticks(np.arange(n_nodes) + 0.5)  # Adjust to center of the grid cell
        axmatrix.set_yticklabels(labels, fontsize=8)
    else:  # if nodes are the columns, show labels in x-axis (top)
        axmatrix.set_xlim([-1.5, n_nodes - 0.5])
        axmatrix.set_xticks(np.arange(n_nodes) + 0.5)  # Adjust to center of the grid cell
        axmatrix.set_xticklabels(labels, rotation=90, fontsize=8)

    # Add colored rectangles for regions
    for i in range(n_nodes):
        r = areas[i]
        color = color_regions.loc[color_regions.Area == r, 'color'].values[0]
        if orientation == 0:  # if nodes are in rows, show rectangles in y-axis (left)
            axmatrix.add_patch(Rectangle((-1.5, i - 0.5), 1, 1, alpha=1, facecolor=color))
        else:  # if nodes are in columns, show rectangles in x-axis (top)
            axmatrix.add_patch(Rectangle((i - 0.5, n_nodes - 0.5), 1, 1, alpha=1, facecolor=color))

    if axcolor:
        plt.colorbar(im, cax=axcolor)


def plot_block_hypnogram(M, hypnogram, x_ax_h, x_ax, x_ax_block, h_diff=12):
    from matplotlib.gridspec import GridSpec

    # Convert values to hour:min format while ensuring they are less than 24
    x_ticks_labels = []
    x_ticks_positions = []

    value0 = x_ax_h[0] - h_diff

    for i, value in enumerate(x_ax_h):
        while value > 24:
            value -= 24  # Subtract 24 until it's less than 24

        if (value - value0 >= h_diff) or ((value + 24 - value0 >= h_diff) and (value < value0)):
            x_ticks_labels.append(f"{int(value):02d}:{int((value % 1) * 60):02d}")
            x_ticks_positions.append(x_ax_h[i])
            value0 = value

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=[5, 0.5], height_ratios=[1, 5], wspace=0.2, hspace=0.2)

    # Subplot for the correlation matrix
    ax = plt.subplot(gs[1, 0])
    # ax.set_aspect('equal')
    ax_h = plt.subplot(gs[0, 0], sharex=ax)
    ax_cbar = plt.subplot(gs[1, 1])

    ax_h.plot(x_ax_h, hypnogram, c='black', linewidth=2)
    ax_h.axhspan(-1, 0.2, color=color_elab[0, :])  # Using color map for color
    ax_h.fill_between(x_ax_h, hypnogram, -1, color=color_elab[0, :])  # Using color map for color
    ax_h.set_yticks([0, 1, 2, 3, 4])
    ax_h.set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
    ax_h.set_ylim([-1, 5])
    ax_h.invert_yaxis()
    #
    ax_h.set_xticks(x_ticks_positions)
    ax_h.set_xticklabels(x_ticks_labels)  # , rotation=45, fontsize=8
    ax_h.set_ylabel('Score')
    ax_h.set_xlim(x_ax[0], x_ax[-1])

    # Plot Pearson correlation matrix (M)
    im = ax.pcolormesh(x_ax, x_ax_block, M, cmap='jet', vmin=np.percentile(M, 10), vmax=np.percentile(M, 90))
    ax.set_ylabel('Block Number')
    ax.set_xlabel('Block Number')
    ax.set_xlim(x_ax[0], x_ax[-1])
    # Add a colorbar to the last subplot
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.set_ylabel('Pearsons Correlation')
    plt.tight_layout()
    return fig
