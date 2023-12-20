import start_cut_resp
import start_BM_blocks as BM_blocks
import concat
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append('/py_functions')
from glob import glob
import basic_func as bf
from matplotlib.patches import Rectangle
from pathlib import Path
import LL_funcs as LLf
import freq_funcs as ff
import h5py
import significance_funcs as sf
import copy
import tqdm

dist_groups = np.array([[0, 30], [30, 60], [60, 120]])
dist_labels = ['local (<30 mm)', 'short (<60mm)', 'long']
Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

### Preparation
subjs = ["EL028"]
get_data = 1
if get_data:
    for subj in subjs:
        ### cut data in epochs: still in .npy
        start_cut_resp.compute_cut(subj, skip_exist=0, prots=['BM', 'IO'])

        ### get con_trial --for each conenction and trial, save LL value
        BM_blocks.cal_con_trial(subj, cond_folder='CR', skip_block=0, skip_single=0)
        ### concatenates all epoched data into one large h5py file, all responses accessible
        for f in ['BrainMapping']:
            concat.concat_resp_condition(subj, folder=f, cond_folder='CR', skip=0)


#################### BM Analysis
##function to plot BM
def plot_BM_CR_trial_sig(M, labels, areas, label, t):
    cmap_LL = 'hot'
    M[np.isnan(M)] = -1

    time = str(t).zfill(2) + ':00'
    fig = plt.figure(figsize=(25, 25))

    cmap = copy.copy(plt.cm.get_cmap(cmap_LL))  # cmap = copy.copy(mpl.cm.get_cmap(cmap))
    cmap.set_under('black')
    cmap.set_bad('black')
    M = np.ma.masked_equal(M, 0)

    axmatrix = fig.add_axes([0.15, 0.15, 0.7, 0.7])  # x, y, (start posiion), lenx, leny
    im = axmatrix.matshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=8)  # np.percentile(M, 95)
    plt.xlim([-1.5, M.shape[0] - 0.5])
    plt.ylim([-0.5, M.shape[1] + 0.5])
    plt.xticks(range(M.shape[1]), labels[:M.shape[1]], rotation=90);
    plt.yticks(range(M.shape[0]), labels[:M.shape[0]]);
    for i in range(len(labels)):
        r = areas[i]
        if i <= M.shape[1]:
            axmatrix.add_patch(Rectangle((i - 0.5, len(labels) - 0.5), 1, 1, alpha=1,
                                         facecolor=color_regions[np.where(regions == r)[0][0]]))
        if i <= M.shape[0]:
            axmatrix.add_patch(
                Rectangle((-1.5, i - 0.5), 1, 1, alpha=1, facecolor=color_regions[np.where(regions == r)[0][0]]))
    # Plot colorbar.
    # axcolor = fig.add_axes([0.04,0.85,0.08,0.08]) # x, y, x_len, y_len
    # circle1 = plt.Circle((0.5,0.5), 0.4, color = CR_color[t], alpha = CR_color_a[t])
    # plt.text(0.3,0.3, time)
    # plt.axis('off')
    # axcolor.add_patch(circle1)
    axcolor = fig.add_axes([0.9, 0.15, 0.01, 0.7])  # x, y, x_len, y_len
    plt.colorbar(im, cax=axcolor)
    plt.title(label + ', ' + time + '-- Sig')
    plt.savefig(path_patient_analysis + '/BrainMapping/CR/figures/BM_plot/BM_' + label + '.png', dpi=300)
    Path(path_patient + '/Analysis/' + folder + '/' + cond_folder + '/figures/').mkdir(parents=True, exist_ok=True)
    # plt.savefig(path_patient + '/Analysis/' + folder + '/' + cond_folder +'/figures/'+'BM_clin.png', )
    plt.show()
    # plt.close(fig) #plt.show()#

def sort_areas(areas, regions):
    # Create a mapping of regions to their index for sorting
    region_order = {region: index for index, region in enumerate(regions)}
    # Define the sorting key
    def get_sort_key(area):
        # Split the label into the side (L_/R_) and the region name
        side, region_name = area.split("_")
        # Normalize the region name to match the case of the regions list
        region_name = region_name.capitalize()
        # Get the index of the region name from the region_order mapping
        region_index = region_order.get(region_name, -1)
        # Return a tuple that Python will use to sort the areas
        return (side, region_index)

    # Get the sort keys for all areas
    sort_keys = [get_sort_key(area) for area in areas]
    # Get the indices for the sorted areas
    indices = sorted(range(len(areas)), key=lambda i: sort_keys[i])
    # Sort the areas using the sorting indices
    sorted_areas = [areas[i] for i in indices]
    return sorted_areas, indices


for subj in subjs:
    cond_folder = 'CR'  # Condition = 'Hour', 'Condition', 'Ph'
    folder = 'BrainMapping'

    ##Patient Analysis path
    path_patient_analysis = sub_path + '\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_gen = os.path.join(sub_path + '\Patients\\' + subj)

    path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    path_infos = os.path.join(path_gen, 'Electrodes')
    if not os.path.exists(os.path.join(path_infos, subj + "_labels.xlsx")):
        path_infos = os.path.join(path_gen, 'infos')

    files_list = glob(path_patient_analysis + '\\' + folder + '/data/Stim_list_*')
    stimlist = pd.read_csv(files_list[0])

    ### Get labels from excel which is stored in "Electrodes" - same that is used to calculate BP
    lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
    if "type" in lbls:
        lbls = lbls[lbls.type == 'SEEG']
        lbls = lbls.reset_index(drop=True)
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist,
        lbls)
    # channel selection
    badchans = pd.read_csv(
        path_patient_analysis + '\\' + folder + '/data/badchan.csv')  # load bad chan if there are any (artifacts)
    bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])
    bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[
        0]  # remove WM, OUT, ...

    # regions = np.unique(labels_region)
    # # very bad hardcoded....
    # color_regions = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
    #                  "#8FB996"]

    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
    tab_region = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='plot')
    tab_region = tab_region.sort_values('Order').reset_index(drop=True)
    regions = tab_region.Area.values
    color_regions = tab_region.color.values

    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
    all_region = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')

    ###### Load data
    file_M = path_patient_analysis + '\\' + folder + '/' + cond_folder + '/data/M_npy'
    file_con = path_patient_analysis + '\\' + folder + '/' + cond_folder + '/data/con_trial_all.csv'
    con_trial = pd.read_csv(file_con)  # for each conenction and trial, save LL value
    # concatenated Epoched data
    h5_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.h5'
    EEG_resp = h5py.File(h5_file)
    EEG_resp = EEG_resp['EEG_resp']

    ## calulate LL for each connection and save if it's higher than threshold
    t_0 = 1  # timepoint in epoched data where stim onset is
    w = 0.25  # window for LL calculation
    stimchans = np.unique(con_trial.Stim).astype('int')  # all channels that were stimualted
    if os.path.exists(path_patient_analysis + '\\' + folder + '/' + cond_folder + '/data/M01.npy'):
        M = np.load(path_patient_analysis + '\\' + folder + '/' + cond_folder + '/data/M01.npy')
    else:
        M = np.zeros((len(labels_all), len(labels_all)))
        for sc in tqdm.tqdm(stimchans):
            for rc in range(len(labels_all)):
                data = con_trial[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1)]
                stimnum = data.Num.values.astype('int')
                if len(stimnum) > 0:
                    resp = np.mean(EEG_resp[rc, stimnum, :], 0)
                    resp = ff.lp_filter(resp, 45, Fs)
                    resp_LL = LLf.get_LL_all(np.expand_dims(resp, [0, 1]), Fs, w)[0][0]

                    thr = np.percentile(np.concatenate([resp_LL[int((w / 2) * Fs):int((t_0 - w / 2) * Fs)],
                                                        resp_LL[int(3 * Fs):int((4 - w / 2) * Fs)]]),
                                        99)  # LL_resp[0, 0, int((t_0+0.5) * Fs):] = 0 * Fs):] = 0
                    LL_t = np.array(resp_LL[int((t_0 - w / 2) * Fs):int((t_0 + 0.5 - w / 2) * Fs)] > thr) * 1
                    t_resp_all = sf.search_sequence_numpy(LL_t, np.ones((int((w) * Fs),)))

                    if len(t_resp_all) > 0:
                        M[sc, rc] = np.nanmax(resp_LL[int((t_0 + w / 2) * Fs):int((t_0 + 0.3 + w / 2) * Fs)])
                else:
                    M[sc, rc] = np.nan
        np.save(path_patient_analysis + '\\' + folder + '/' + cond_folder + '/data/M01.npy', M)
    StimChanIx = np.array(StimChanIx).astype('int')
    ## remove bad channels from Matrix
    non_stim = np.arange(len(labels_all))
    non_stim = np.delete(non_stim, StimChanIx, 0)
    WM_chans = np.where(labels_region == 'WM')[0]
    bad_all = np.unique(np.concatenate([WM_chans, bad_region, bad_chans, non_stim])).astype('int')
    M_resp = np.delete(np.delete(M, bad_all, 0), bad_all, 1)

    ## select labels without bad channels
    hem = lbls.Hemisphere.values
    labels_clin = np.delete(labels_clinic, bad_all, 0)
    areas_sel = np.delete(labels_region, bad_all, 0)
    areas_sel_sort = np.delete(hem + '_' + labels_region, bad_all, 0)
    labels_sel = np.delete(labels_all, bad_all, 0)
    labels_sel = labels_sel + ' (' + labels_clin + ')'
    order_anat = 1
    ll = 'H_clinic'
    if order_anat:
        sorted_areas, ind = sort_areas(areas_sel_sort, regions)
        labels_sel = labels_sel[ind]
        areas_sel = areas_sel[ind]
        M_resp = M_resp[ind, :]
        M_resp = M_resp[:, ind]
        ll = 'H_anat'

    plot_BM_CR_trial_sig(M_resp, labels_sel, areas_sel, ll, 't')
    print('Stop')


# Correcting the sorting function

