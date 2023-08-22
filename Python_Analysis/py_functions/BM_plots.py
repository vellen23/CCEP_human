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

regions = pd.read_excel("T:\EL_experiment\Patients\\all\elab_labels.xlsx", sheet_name='regions', header=0)

color_regions = regions.color.values
regions_G = regions.subregion.values
regions = regions.label.values

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]

# fig = plt.figure(figsize=(25, 25))
# fig.patch.set_facecolor('xkcd:white')
# axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])  # x, y, (start posiion), lenx, leny
CIRC_AREAS_FILEPATH = "X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\\circ_areas.xlsx"
color_regions = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='plot')
atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')


def clean_label(labels):
    """Remove spaces and numbers from labels."""
    labels = [label.replace(" ", "") for label in labels]
    labels = [re.sub(r'\d', '', label) for label in labels]
    return labels


def remove_bad_areas(M, labels, regions, hemisphere):
    """Filter out bad regions from the matrix and labels."""
    BAD_REGIONS = {'WM', 'OUT', 'out', 'Putamen', 'Pallidum', 'Necrosis', 'LV'}

    valid_indices = [i for i, region in enumerate(regions) if region not in BAD_REGIONS]
    M = M[valid_indices][:, valid_indices]
    labels = [labels[i] for i in valid_indices]
    regions = [regions[i] for i in valid_indices]
    hemisphere = [hemisphere[i] for i in valid_indices]
    return M, labels, regions, hemisphere


def get_region(labels, atlas):
    """Retrieve the region for each label from the atlas."""
    return [atlas.loc[atlas.Abbreviation == l, 'Region'].values[0] if l in atlas.Abbreviation.values else 'Unknown' for
            l in labels]


def sort_M(M, labels, regions, hemisphere):
    areas_sel_sort = [h + '_' + r for h, r in zip(hemisphere, regions)]
    ind = np.argsort(areas_sel_sort)

    M = M[ind][:, ind]
    labels = [labels[i] for i in ind]
    regions = [regions[i] for i in ind]
    hemisphere = [hemisphere[i] for i in ind]

    return M, labels, regions


def plot_BM(M, labels, hemisphere, axmatrix, axcolor=None, cmap='hot', vlim=None, sort=1):
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
        plt.colorbar(im, cax=axcolor)