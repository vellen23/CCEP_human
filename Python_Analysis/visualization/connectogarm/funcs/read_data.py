from threading import *
import wx
import sys
import time
import numpy as np
import os
import pandas as pd
import plot_funcs as pf


def get_nodes(chan_ID, data_con):
    data_nodes = pd.DataFrame(chan_ID.T, columns=['ID'])
    data_nodes.insert(0, 'Label', 'Test')
    data_nodes.insert(0, 'Region', 'Test')
    data_nodes.insert(0, 'Subj', 'Test')

    for c in chan_ID:
        data_nodes.loc[data_nodes.ID == c, ['Label', 'Region', 'Subj']] = data_con.loc[
                                                                              (data_con.Stim == c), ['StimA', 'StimR',
                                                                                                     'Subj']].values[0,
                                                                          :]
        data_nodes.loc[data_nodes.ID == c, ['Label', 'Region', 'Subj']] = data_con.loc[
                                                                              (data_con.Chan == c), ['ChanA', 'ChanR',
                                                                                                     'Subj']].values[0,
                                                                          :]

    data_nodes.insert(4, 'theta', 0)
    data_nodes.insert(4, 'y', 0)
    data_nodes.insert(4, 'x', 0)
    data_nodes.loc[data_nodes.Region == 'Sylvian', 'Region'] = 'Central'
    data_nodes = data_nodes[data_nodes.Region != 'Unknown']
    data_nodes = data_nodes.reset_index(drop=True)
    return data_nodes


def get_info_c(areas_c, n_nodes, radius, plot_hem='r'):
    for region in areas_c.Area:
        areas_c.loc[areas_c.Area == region, 'N_nodes'] = n_nodes[region]
    areas_c.loc[np.isnan(areas_c.N_nodes), 'N_nodes'] = np.nanmin(areas_c.N_nodes)

    #
    ratios = areas_c.N_nodes.values
    ratios = ratios / np.min(ratios)
    # arr = np.array([8,32,36])
    # result = np.gcd.reduce(arr)
    ratios_n = np.ones((len(ratios),))
    ratios_n[ratios > np.percentile(ratios, 33)] = 2
    ratios_n[ratios > np.percentile(ratios, 66)] = 3
    # ratios_n = ratios
    tot_seg = np.sum(ratios_n)
    start = np.pi / 2
    n_areas_c = len(areas_c)
    area_c_borders = np.zeros((n_areas_c, 2))
    gap = np.pi / tot_seg / 10
    for i in range(n_areas_c):
        area_c_borders[i, 0] = start - gap  # (np.pi/2 - i * np.pi / n_areas)-0.01
        area_c_borders[i, 1] = start + gap - np.pi * ratios_n[i] / tot_seg  # (np.pi/2 - (i+1) * np.pi / n_areas)+0.01
        start = start - np.pi * ratios_n[i] / tot_seg
    # area_c_borders[area_c_borders>np.pi] = area_c_borders[area_c_borders>np.pi] -np.pi
    areas_c.insert(5, 'theta1', area_c_borders[:, 1])
    areas_c.insert(5, 'theta0', area_c_borders[:, 0])

    areas_c_xy = np.zeros((n_areas_c, 4))
    for i in range(n_areas_c):
        areas_c_xy[i, :2] = pf.to_cartesian(r=radius, theta=areas_c.theta0.values[i])  # area_borders[i,1]
        areas_c_xy[i, 2:] = pf.to_cartesian(r=radius, theta=areas_c.theta1.values[i])
    areas_c.insert(5, 'y1', areas_c_xy[:, 3])
    areas_c.insert(5, 'y0', areas_c_xy[:, 1])
    areas_c.insert(5, 'x1', areas_c_xy[:, 2])
    areas_c.insert(5, 'x0', areas_c_xy[:, 0])

    if plot_hem == 'l':
        new_x1 = -areas_c.x1.values
        new_x0 = -areas_c.x0.values
        areas_c.x0 = new_x0
        areas_c.x1 = new_x1
        for i in range(len(areas_c)):
            r, areas_c.theta1.values[i] = pf.to_polar(areas_c.x0.values[i], areas_c.y0.values[i], theta_units="radians")
            r, areas_c.theta0.values[i] = pf.to_polar(areas_c.x1.values[i], areas_c.y1.values[i], theta_units="radians")
    return areas_c


def get_info_s(areas_s, radius, plot_hem='r'):
    n_areas_s = len(areas_s)
    if plot_hem == 'r':
        x = -0.2 * radius
    else:
        x = +0.2 * radius
    y_start = radius - 0.1 * radius
    y_end = -radius + 0.1 * radius
    y_lin = np.linspace(y_start, y_end, n_areas_s + 1)
    l_s = 0.9 * abs(y_lin[0] - y_lin[1])
    areas_s_xy = np.zeros((n_areas_s, 2, 2))
    areas_s_xy[:, :, 0] = x
    areas_s_xy[:, 0, 1] = y_lin[:-1]  # *1.1
    areas_s_xy[:, 1, 1] = y_lin[:-1] - l_s  # y_lin[1:]*0.9
    areas_s.insert(5, 'y1', areas_s_xy[:, 1, 1])
    areas_s.insert(5, 'y0', areas_s_xy[:, 0, 1])
    areas_s.insert(5, 'x1', areas_s_xy[:, 1, 0])
    areas_s.insert(5, 'x0', areas_s_xy[:, 0, 0])
    areas_s = areas_s.sort_values(by=['Order'])
    areas_s = areas_s.reset_index(drop=True)
    return areas_s, l_s


def get_node_coord(data_nodes, areas_c, areas_s, r_nodes, l_s):
    for region in areas_c.Area:
        n_node = len(data_nodes[data_nodes.Region == region])
        if n_node > 0:
            t = areas_c.loc[areas_c.Area == region, ['theta0', 'theta1']].values[0]  # both theta
            d0 = 0
            if (t[0] < 0) & (t[1] > 0):
                d0 = 2 * np.pi
            data_nodes.loc[data_nodes.Region == region, 'theta'] = np.linspace(t[1], t[0] + d0, n_node)
            data_nodes.loc[data_nodes.theta > np.pi, 'theta'] = data_nodes.loc[
                                                                    data_nodes.theta > np.pi, 'theta'].values - 2 * np.pi
            xy = np.zeros((n_node, 2))
            for i in range(n_node):
                x, y = pf.to_cartesian(r=r_nodes, theta=data_nodes.loc[data_nodes.Region == region, 'theta'].values[i])
                xy[i, 0] = x
                xy[i, 1] = y
            data_nodes.loc[data_nodes.Region == region, 'x'] = xy[:, 0]
            data_nodes.loc[data_nodes.Region == region, 'y'] = xy[:, 1]

    for region in areas_s.Area:
        n_node = len(data_nodes[data_nodes.Region == region])
        if n_node > 0:
            y = areas_s.loc[areas_s.Area == region, ['y0', 'y1']].values[0]
            xy = np.zeros((n_node, 2))
            xy[:, 0] = areas_s.loc[areas_s.Area == region, ['x0']].values[0][0]
            xy[:, 1] = np.linspace(y[0], y[0] - l_s, n_node)

            data_nodes.loc[data_nodes.Region == region, 'x'] = xy[:, 0]
            data_nodes.loc[data_nodes.Region == region, 'y'] = xy[:, 1]

    return data_nodes
