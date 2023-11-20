import os
import numpy as np
import scipy
import tqdm
import pandas as pd
import sys
import re

sys.path.append('./py_functions')
import scipy.io
import matplotlib.cm as cm

# import matplotlib.colors as mcolors
folder = 'BrainMapping'
cond_folder = 'CR'
dist_groups = np.array([[0, 15], [15, 30], [30, 5000]])
dist_labels = ['local (<15 mm)', 'short (<30mm)', 'long']


def get_color(group='Dist'):
    cmap_org = 'winter'
    n = 45
    viridis = cm.get_cmap(cmap_org, n)
    viridis_org = viridis(np.linspace(0, 1, n))
    color_b = np.array([viridis_org[0], viridis_org[15], viridis_org[30], viridis_org[-1]])

    n = 100
    viridis = cm.get_cmap(cmap_org, n)
    color_d = viridis(np.linspace(0, 1, n))

    inter_col = np.zeros((15, 4))
    for i in range(4):
        inter_col[:, i] = np.linspace(color_b[0, i], color_b[1, i], 15)

    # 1.
    inter_col = np.zeros((15, 4))
    for i in range(4):
        inter_col[:, i] = np.linspace(color_b[0, i], color_b[1, i], 15)

    color_d[0:15, :] = inter_col

    # 2.
    inter_col = np.zeros((15, 4))
    for i in range(4):
        inter_col[:, i] = np.linspace(color_b[1, i], color_b[2, i], 15)

    color_d[15:30, :] = inter_col

    # 3.
    inter_col = np.zeros((70, 4))
    for i in range(4):
        inter_col[:, i] = np.linspace(color_b[2, i], color_b[3, i], 70)

    color_d[30:100, :] = inter_col

    color_dist = np.array([color_d[0], color_d[20], color_d[70]])
    color_group = np.zeros((3, 3))
    color_group[0, :] = np.array([241, 93, 93]) / 255
    color_group[1, :] = np.array([157, 191, 217]) / 255
    color_group[2, :] = np.array([162, 209, 164]) / 255

    color_elab = np.zeros((4, 3))
    color_elab[0, :] = np.array([31, 78, 121]) / 255
    color_elab[1, :] = np.array([189, 215, 28]) / 255
    color_elab[2, :] = np.array([0.256, 0.574, 0.431])
    color_elab[3, :] = np.array([1, 0.574, 0])

    return color_d, color_dist, color_group, color_elab


def adding_area(data_A, lbls, pair=1):
    labels_all = lbls.label.values
    if pair:
        for c in np.unique(data_A[['Chan', 'Stim']]).astype('int'):
            data_A.loc[data_A.Chan == c, 'ChanA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
            data_A.loc[data_A.Stim == c, 'StimA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
            chans = data_A.loc[data_A.Stim == c, 'Chan'].values.astype('int')
            data_A.loc[data_A.Stim == c, 'H'] = np.array(lbls.Hemisphere[chans] != lbls.Hemisphere[c]) * 1
    else:
        for c in np.unique(data_A[['Chan']]).astype('int'):
            data_A.loc[data_A.Chan == c, 'ChanA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
    return data_A


def adding_subregion(data_con, pair=1, area='Subregion'):
    # area == 'Region' or 'Area'
    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
    atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')
    if pair:
        for subregion in np.unique(data_con[['StimA', 'ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values
            if len(region) > 0:
                data_con.loc[data_con.StimA == subregion, 'StimSR'] = region[0]
                data_con.loc[data_con.ChanA == subregion, 'ChanSR'] = region[0]
            else:
                print(subregion)
                data_con.loc[data_con.StimA == subregion, 'StimSR'] = 'U'
                data_con.loc[data_con.ChanA == subregion, 'ChanSR'] = 'U'
    else:
        for subregion in np.unique(data_con[['ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values
            if len(region) > 0:
                data_con.loc[data_con.ChanA == subregion, 'ChanSR'] = region[0]
            else:
                data_con.loc[data_con.ChanA == subregion, 'ChanSR'] = 'U'
    return data_con


def adding_region(data_con, pair=1, area='Region'):
    # area == 'Region' or 'Area'
    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
    atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')
    if pair:
        for subregion in np.unique(data_con[['StimA', 'ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values
            if len(region) > 0:
                data_con.loc[data_con.StimA == subregion, 'StimR'] = region[0]
                data_con.loc[data_con.ChanA == subregion, 'ChanR'] = region[0]
            else:
                print(subregion)
                data_con.loc[data_con.StimA == subregion, 'StimR'] = 'U'
                data_con.loc[data_con.ChanA == subregion, 'ChanR'] = 'U'
    else:
        for subregion in np.unique(data_con[['ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values
            if len(region) > 0:
                data_con.loc[data_con.ChanA == subregion, 'ChanR'] = region[0]
            else:
                data_con.loc[data_con.ChanA == subregion, 'ChanR'] = 'U'
    return data_con


def get_DI(subjs, sub_path, filename):
    # Initialize an empty DataFrame to store the results
    data_con = pd.DataFrame()
    for i in range(len(subjs)):
        print('loading -- ' + subjs[i], end='\r')
        subj = subjs[i]
        path_gen = os.path.join(sub_path + 'Patients\\' + subj)
        if not os.path.exists(path_gen):
            path_gen = 'T:\\EL_experiment\\Patients\\' + subj
        path_patient = path_gen + '\Data\EL_experiment'
        path_infos = os.path.join(path_patient, 'infos')
        if not os.path.exists(path_infos):
            path_infos = path_gen + '\\infos'
        path_patient_analysis = sub_path + '\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
        summary_gen_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\M_DI.csv'
        data_A = pd.read_csv(summary_gen_path)

        lbls = pd.read_excel(os.path.join(path_gen, 'Electrodes', subj + "_labels.xlsx"), header=0, sheet_name='BP')
        labels_all = lbls.label.values
        labels_region = lbls.Region.values

        bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]

        StimChanIx = np.unique(data_A.A)
        bad_chans = pd.read_csv(path_patient_analysis + '/BrainMapping/data/badchan.csv')
        bad_chans = np.unique(np.array(np.where(bad_chans.values[:, 1:] == 1))[0, :])
        non_stim = np.arange(len(labels_all))
        non_stim = np.delete(non_stim, StimChanIx.astype('int'), 0)
        WM_chans = np.where(labels_region == 'WM')[0]
        bad_all = np.unique(np.concatenate([WM_chans, bad_region, bad_chans, non_stim])).astype('int')
        coord = lbls[['x', 'y', 'z']]

        data_A = data_A[~np.isin(data_A.A, bad_all) & ~np.isin(data_A.B, bad_all)]
        data_A.reset_index(drop=True)
        data_A.insert(0, 'Subj', subjs[i])
        data_A.insert(1, 'A_Area', '0')
        data_A.insert(2, 'B_Area', '0')
        data_A.insert(8, 'H', 0)
        data_A.insert(6, 'd', 0)
        for c in np.unique(data_A[['A', 'B']]).astype('int'):
            data_A.loc[data_A.A == c, 'A_Area'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
            data_A.loc[data_A.B == c, 'B_Area'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
            chans = data_A.loc[data_A.A == c, 'B'].values.astype('int')
            data_A.loc[data_A.A == c, 'H'] = np.array(lbls.Hemisphere[chans] != lbls.Hemisphere[c]) * 1
            for rc in chans:
                data_A.loc[(data_A.B == rc) & (data_A.A == c), 'd'] = np.round(
                    scipy.spatial.distance.euclidean(coord.values[c, :], coord.values[rc, :]), 2)

        # Concatenate current data with the accumulated results
        data_con = pd.concat([data_con, data_A], ignore_index=True)

    data_con.loc[data_con.P_AB == -1, 'P_AB'] = np.nan
    data_con.loc[data_con.P_BA == -1, 'P_BA'] = np.nan

    data_con.insert(4, 'DI_d', data_con.P_AB - data_con.P_BA)
    data_con.insert(4, 'DI_r', 1 - (np.min([data_con.P_AB.values, data_con.P_BA.values], 0) / np.max(
        [data_con.P_AB.values, data_con.P_BA.values], 0)))
    data_con.DI_r = data_con.DI_r * np.sign(data_con.DI_d)

    data_con.loc[(data_con.P_BA == 0) & (data_con.P_AB > 0), 'DI_d'] = 1
    data_con.loc[(data_con.P_AB == 0) & (data_con.P_BA > 0), 'DI_d'] = -1
    data_con.loc[(data_con.P_AB == 0) & (data_con.P_BA == 0), 'DI_d'] = np.nan

    data_con.insert(0, 'Dist', dist_labels[-1])
    data_con.loc[data_con.d <= 30, 'Dist'] = dist_labels[1]
    data_con.loc[data_con.d <= 15, 'Dist'] = dist_labels[0]
    col_drop = ['Dist', 'A_Area', 'A_Region', 'B_Area', 'B_Region', 'DI_r', 'DI_d']
    for col in col_drop:
        if col in data_con:
            data_con = data_con.drop(columns=[col])
    data_con.to_csv(filename, header=True, index=False)


def get_connections(subjs, sub_path, filename):
    path_export = os.path.join(sub_path,
                               'EvM\\Projects\\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\\')
    data_con_all = pd.DataFrame()
    for i in range(len(subjs)):
        print('loading -- ' + subjs[i], end='\r')
        subj = subjs[i]
        path_gen = os.path.join(sub_path + 'Patients\\' + subj)
        if not os.path.exists(path_gen):
            path_gen = 'T:\\EL_experiment\\Patients\\' + subj
        path_patient = path_gen + '\Data\EL_experiment'
        path_infos = os.path.join(path_patient, 'infos')
        if not os.path.exists(path_infos):
            path_infos = path_gen + '\\infos'

        path_patient_analysis = sub_path + '\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
        summary_gen_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\' + filename
        data_A = pd.read_csv(summary_gen_path)

        lbls = pd.read_excel(os.path.join(path_gen, 'Electrodes', subj + "_labels.xlsx"), header=0, sheet_name='BP')
        labels_all = lbls.label.values
        labels_clinic = lbls.Clinic.values
        labels_region = lbls.Region.values

        bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]

        StimChanIx = np.unique(data_A.Stim)
        bad_chans = pd.read_csv(path_patient_analysis + '/BrainMapping/data/badchan.csv')
        bad_chans = np.unique(np.array(np.where(bad_chans.values[:, 1:] == 1))[0, :])
        non_stim = np.arange(len(labels_all))
        non_stim = np.delete(non_stim, StimChanIx.astype('int'), 0)
        WM_chans = np.where(labels_region == 'WM')[0]
        bad_all = np.unique(np.concatenate([WM_chans, bad_region, bad_chans, non_stim])).astype('int')

        data_A = data_A[~np.isin(data_A.Chan, bad_all) & ~np.isin(data_A.Stim, bad_all)]
        data_A.reset_index(drop=True)
        data_A.insert(0, 'Subj', subjs[i])
        data_A.insert(1, 'StimA', '0')
        data_A.insert(2, 'ChanA', '0')
        data_A.insert(8, 'H', 0)
        for c in np.unique(data_A[['Chan', 'Stim']]).astype('int'):
            data_A.loc[data_A.Chan == c, 'ChanA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
            data_A.loc[data_A.Stim == c, 'StimA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
            chans = data_A.loc[data_A.Stim == c, 'Chan'].values.astype('int')
            data_A.loc[data_A.Stim == c, 'H'] = np.array(lbls.Hemisphere[chans] != lbls.Hemisphere[c]) * 1

        # Concatenate current data with the accumulated results
        data_con_all = pd.concat([data_con_all, data_A], ignore_index=True)
    data_con_all.Stim = data_con_all.Stim.astype('int')
    data_con_all.Chan = data_con_all.Chan.astype('int')
    data_con_all = adding_subregion(data_con_all)
    data_con_all = adding_region(data_con_all)
    data_con_all.to_csv(os.path.join(path_export, filename), header=True, index=False)


def get_connections_sleep(subjs, sub_path, filename):
    chan_n_max = 0
    for i in range(len(subjs)):
        print('loading -- ' + subjs[i], end='\r')
        subj = subjs[i]
        path_gen = os.path.join(sub_path + '\Patients\\' + subj)
        if not os.path.exists(path_gen):
            path_gen = 'T:\\EL_experiment\\Patients\\' + subj
        path_patient = path_gen + '\Data\EL_experiment'
        path_infos = os.path.join(path_patient, 'infos')
        if not os.path.exists(path_infos):
            path_infos = path_gen + '\\infos'
        path_patient_analysis = sub_path + '\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj

        file_con_sleep = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_sleep_stats.csv'
        # , con_sleep_stats, con_sleep
        if os.path.exists(file_con_sleep):
            data_A = pd.read_csv(file_con_sleep)

            lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
            labels_all = lbls.label.values
            labels_region = lbls.Region.values
            bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]

            StimChanIx = np.unique(data_A.Stim)
            bad_chans = pd.read_csv(path_patient_analysis + '/BrainMapping/data/badchan.csv')
            bad_chans = np.unique(np.array(np.where(bad_chans.values[:, 1:] == 1))[0, :])
            non_stim = np.arange(len(labels_all))
            non_stim = np.delete(non_stim, StimChanIx.astype('int'), 0)
            WM_chans = np.where(labels_region == 'WM')[0]
            bad_all = np.unique(np.concatenate([WM_chans, bad_region, bad_chans, non_stim])).astype('int')

            data_A = data_A[~np.isin(data_A.Chan, bad_all) & ~np.isin(data_A.Stim, bad_all)]
            data_A.reset_index(drop=True)
            data_A.insert(0, 'Subj', subjs[i])
            data_A.insert(1, 'StimA', '0')
            data_A.insert(2, 'ChanA', '0')
            data_A.insert(1, 'Stim_ID', data_A.Stim)
            data_A.insert(2, 'Chan_ID', data_A.Chan)
            data_A.insert(8, 'H', 0)
            for c in np.unique(data_A[['Chan', 'Stim']]).astype('int'):
                data_A.loc[data_A.Chan == c, 'ChanA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
                data_A.loc[data_A.Stim == c, 'StimA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
                chans = data_A.loc[data_A.Stim == c, 'Chan'].values.astype('int')
                data_A.loc[data_A.Stim == c, 'H'] = np.array(lbls.Hemisphere[chans] != lbls.Hemisphere[c]) * 1

            # data_A = data_A[~np.isnan(data_A.N1.values)]
            data_A.Stim_ID = data_A.Stim + chan_n_max
            data_A.Chan_ID = data_A.Chan + chan_n_max

            if chan_n_max == 0:
                data_con = data_A
            else:
                data_con = pd.concat([data_con, data_A])
                data_con = data_con.reset_index(drop=True)
            chan_n_max = np.max(data_con[['Stim_ID', 'Chan_ID']].values) + 1
    data_con = data_con[(data_con.ChanA != 'Necrosis') & (data_con.StimA != 'Necrosis')]

    data_con.to_csv(filename,
                    header=True, index=False)


def update_DI(data_con_all):
    col = ['Subj', 'A', 'B', 'P_AB', 'P_BA', 'LL_A', 'LL_B', 'onset_A',
           'onset_B', 'd', 'DI']

    data_DI2 = pd.DataFrame(columns=col)  # Initialize an empty DataFrame to store results

    unique_subjs = np.unique(data_con_all.Subj)

    for subj in tqdm.tqdm(unique_subjs):
        subj_data = data_con_all[data_con_all.Subj == subj]
        chans = np.unique(subj_data[['Stim', 'Chan']]).astype('int')

        for sc in chans:
            sc_data = subj_data[(subj_data.Stim == sc) & (subj_data.Sig > 0)]

            for rc in chans:
                if sc <= rc:  # Avoid duplicate entries
                    continue

                rc_data = subj_data[(subj_data.Stim == rc) & (rc_data.Sig > 0)]
                d = subj_data[(subj_data.Stim == sc) & (subj_data.Chan == rc)].d.values[0]

                if len(sc_data) > 0 and len(rc_data) > 0:
                    arr = [[subj, sc, rc, sc_data.Sig.values[0], rc_data.Sig.values[0], sc_data.LL_sig.values[0],
                            rc_data.LL_sig.values[0], sc_data.onset.values[0], rc_data.onset.values[0],
                            d, sc_data.DI.values[0]]]
                elif len(sc_data) > 0:
                    arr = [[subj, sc, rc, sc_data.Sig.values[0], 0, sc_data.LL_sig.values[0], np.nan,
                            sc_data.onset.values[0], np.nan, d, 1]]
                elif len(rc_data) > 0:
                    arr = [[subj, sc, rc, 0, rc_data.Sig.values[0], np.nan, rc_data.LL_sig.values[0], np.nan,
                            rc_data.onset.values[0], d, -1]]
                else:
                    arr = [[subj, sc, rc, 0, 0, np.nan, np.nan, np.nan, np.nan, d, np.nan]]

                data_DI2 = data_DI2.append(pd.DataFrame(arr, columns=col), ignore_index=True)

    return data_DI2


def update_DI0(data_con_all):
    # adds t_onset for both directions in a very non-efficient way
    start = 1
    col = ['Subj', 'A', 'B', 'P_AB', 'P_BA', 'LL_A', 'LL_B', 'onset_A',
           'onset_B', 'd', 'DI']
    for subj in tqdm.tqdm(np.unique(data_con_all.Subj)):
        chans = np.unique(data_con_all.loc[data_con_all.Subj == subj, ['Stim', 'Chan']]).astype('int')
        for i in range(len(chans)):
            sc = chans[i]
            for j in range(i, len(chans)):
                rc = chans[j]
                d = data_con_all.loc[
                    (data_con_all.Stim == sc) & (data_con_all.Chan == rc) & (data_con_all.Subj == subj), 'd']
                if len(d) > 0:
                    d = d.values[0]
                    rc = chans[j]
                    data_A = data_con_all.loc[
                        (data_con_all.Sig > 0) & (data_con_all.Stim == sc) & (data_con_all.Chan == rc) & (
                                data_con_all.Subj == subj)]
                    data_B = data_con_all.loc[
                        (data_con_all.Sig > 0) & (data_con_all.Stim == rc) & (data_con_all.Chan == sc) & (
                                data_con_all.Subj == subj)]
                    if (len(data_A) > 0) & (len(data_B) > 0):
                        arr = [[subj, sc, rc, data_A.Sig.values[0], data_B.Sig.values[0], data_A.LL_sig.values[0],
                                data_B.LL_sig.values[0], data_A.onset.values[0], data_B.onset.values[0],
                                data_A.d.values[0], data_A.DI.values[0]]]
                        arr = pd.DataFrame(arr, columns=col)
                    elif (len(data_A) > 0):
                        arr = [[subj, sc, rc, data_A.Sig.values[0], 0, data_A.LL_sig.values[0], np.nan,
                                data_A.onset.values[0], np.nan, data_A.d.values[0], 1]]
                        arr = pd.DataFrame(arr, columns=col)
                    elif (len(data_B) > 0):
                        arr = [[subj, sc, rc, 0, data_B.Sig.values[0], np.nan, data_B.LL_sig.values[0], np.nan,
                                data_B.onset.values[0], data_B.d.values[0], -1]]
                        arr = pd.DataFrame(arr, columns=col)
                    else:
                        arr = [[subj, sc, rc, 0, 0, np.nan, np.nan, np.nan, np.nan, d, np.nan]]
                        arr = pd.DataFrame(arr, columns=col)

                    if start:
                        data_DI2 = arr
                        start = 0
                    else:
                        data_DI2 = pd.concat([data_DI2, arr], 0)
                        data_DI2 = data_DI2.reset_index(drop=True)
    return data_DI2
