from threading import *
import wx
import sys

sys.path.append('./funcs/')

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.path import Path
from more_itertools import unique_everseen
from matplotlib.colors import to_hex
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import itertools
import seaborn as sns

import read_data as rd
import plot_funcs as pf

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def despine(ax):
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


print('start')


class main_plot():
    def __init__(self, data_con, data_nodes, plot_hem='r'):
        """Create the CircosPlot."""
        self.data_con = data_con
        self.data_nodes = data_nodes
        self.plot_hem = plot_hem

        ###
        self.r_seg = 20
        self.radius = 20
        self.r_nodes = 18
        self.load_information()

    def load_information(self):
        circ_areas = pd.read_excel('X:\\4 e-Lab\\EvM\Projects\EL_experiment\Analysis\General_figures\\circ_areas.xlsx') # todo:
        areas_c = circ_areas[circ_areas.Plot == 'c']
        areas_c.insert(5, 'N_nodes', np.nan)
        areas_c = areas_c.sort_values(by=['Order'])
        areas_c = areas_c.reset_index(drop=True)
        n_nodes = self.data_nodes.groupby(['Region'])['ID'].count()
        areas_s = circ_areas[circ_areas.Plot == 's']
        r_seg = 20
        # areas_c = get_info_c(areas_c, r_seg)
        self.areas_s, self.l_s = rd.get_info_s(areas_s, self.r_seg, self.plot_hem)
        self.areas_c = rd.get_info_c(areas_c, n_nodes, self.r_seg, self.plot_hem)

        self.data_nodes = rd.get_node_coord(self.data_nodes, self.areas_c, self.areas_s, self.r_nodes, self.l_s)
        self.circ_areas = circ_areas

    def plot_con(self, data_edges, ax):
        self.data_edges = data_edges
        if ax==0:
            figsize = (40, 40)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
            despine(ax)
        for i in range(len(self.areas_c)):
            t = self.areas_c[['theta0', 'theta1']].values[i]  # both theta
            ring = mpatches.Wedge((0, 0), self.r_seg, math.degrees(t[1]), math.degrees(t[0]),
                                  width=0.05 * self.r_seg, color=self.areas_c.color.values[i])
            ax.add_patch(ring)

            d0 = 0
            if (t[0] < 0) & (t[1] > 0):
                d0 = 2 * np.pi
            t = (t[1] + t[0] + d0) / 2
            if t > np.pi: t = t - 2 * np.pi
            x, y = pf.to_cartesian(r=1.2 * self.r_seg, theta=t)
            ha, va = pf.text_alignment(x, y)
            ax.text(s=self.areas_c.Area.values[i],
                    x=x,
                    y=y,
                    ha=ha,
                    va=va)

        for i in range(len(self.areas_s)):  # n_areas_s = len(areas_s)
            rectangle = mpatches.Rectangle((self.areas_s.x0.values[i], self.areas_s.y0.values[i]),
                                           (-1) ** (np.array(self.plot_hem == 'r')) * 0.05 * self.r_seg, -self.l_s,
                                           color=self.areas_s.color.values[i])
            ax.add_patch(rectangle)
            x = self.areas_s.x0.values[i] - 2 * 0.05 * self.r_seg * (-1) ** (np.array(self.plot_hem == 'l') * 1)
            y = self.areas_s.y1.values[i] + (self.areas_s.y0.values[i] - self.areas_s.y1.values[i]) / 2
            ha, va = pf.text_alignment(y, y)
            ax.text(s=self.areas_s.Area.values[i], x=x, y=y, ha='right', va='center', rotation=90)

        ####Nodes
        go = 1
        for i in range(len(self.data_nodes)):
            # if (-1)**(np.array(plot_hem == 'r')*1)*data_nodes.x.values[i]<0:
            if go:
                node_patch = mpatches.Circle(
                    (self.data_nodes.x.values[i], self.data_nodes.y.values[i]), 0.1, lw=1, zorder=2)
                ax.add_patch(node_patch)
        ####Edges
        for i in range(len(self.data_edges)):  # range(len(data_edges))
            # i = np.random.randint(len(data_edges))
            c0 = self.data_edges.Stim.values[i]
            c1 = self.data_edges.Chan.values[i]
            if (len(self.data_nodes[self.data_nodes.ID == c1]) > 0) & (len(self.data_nodes[self.data_nodes.ID == c0]) > 0):
                r1 = self.data_nodes.loc[self.data_nodes.ID == c0, 'Region'].values[0]
                xy0 = self.data_nodes.loc[self.data_nodes.ID == c0, ['x', 'y']].values[0]
                xy1 = self.data_nodes.loc[self.data_nodes.ID == c1, ['x', 'y']].values[0]
                verts = [xy0,
                         ((-1) ** (np.array(self.plot_hem == 'l') * 1) * self.r_nodes / 3, 0),
                         xy1,
                         ]

                X = np.array(verts)
                n = 100
                tt = np.linspace(0, 1, n)
                xx = pf.P(tt, X)

                ax.plot(xx[:int(n / 2), 0], xx[:int(n / 2), 1], color=[0, 0, 0], alpha=0.3)
                if len(self.circ_areas.loc[self.circ_areas.Area == r1, 'color'].values) > 0:
                    col = self.circ_areas.loc[self.circ_areas.Area == r1, 'color'].values[0]
                else:
                    col = 'k'
                ax.plot(xx[int(n / 2) - 1:, 0], xx[int(n / 2) - 1:, 1], color=col, alpha=0.3)
        if self.plot_hem == 'l':
            ax.set_xlim([-30, 10])
        else:
            ax.set_xlim([-10, 30])
        ax.set_ylim([-30, 30])
        despine(ax)

        plt.show()


data_con_file = sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\data_con_all.csv'
if os.path.exists(data_con_file):
    data_con_all = pd.read_csv(data_con_file)
data_con = data_con_all[~np.isnan(data_con_all.Dir_index)]
data_con = data_con.reset_index(drop=True)
chan_ID = np.unique(np.concatenate([data_con.Stim, data_con.Chan])).astype('int')
data_nodes = rd.get_nodes(chan_ID, data_con)
