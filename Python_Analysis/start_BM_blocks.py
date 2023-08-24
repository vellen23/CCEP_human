import os
import numpy as np
import mne
import imageio
import h5py
# import scipy.fftpack
import matplotlib
import pywt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# from scipy import signal
from matplotlib.colors import ListedColormap
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
from scipy.stats import norm
import LL_funcs
from scipy.stats import norm
from tkinter import filedialog
from tkinter import *
import ntpath
import _thread

root = Tk()
root.withdraw()
import math
import scipy
from scipy import signal
import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import platform
from glob import glob
from scipy.io import savemat

sys.path.append('./PCI/')
sys.path.append('./PCI/PCIst')

import basic_func as bf
from scipy.integrate import simps
from numpy import trapz
import IO_func as IOF
import BM_func as BMf
import tqdm
from matplotlib.patches import Rectangle
from pathlib import Path
import significance_funcs as sig_func
import freq_funcs as ff
# from tqdm.notebook import trange, tqdm
# remove some warnings
import warnings

# I expect to see RuntimeWarnings in this block
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

regions = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\elab_labels.xlsx", sheet_name='regions', header=0)
color_regions = regions.color.values
regions = regions.label.values

CR_color = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\Analysis\BrainMapping\CR_color.xlsx", header=0)
CR_color_a = CR_color.a.values
CR_color = CR_color.c.values
CR_color = np.zeros((24, 3))
CR_color[6:18, :] = np.array([253, 184, 19]) / 255

dist_groups = np.array([[0, 30], [30, 60], [60, 120]])
dist_labels = ['local (<30 mm)', 'short (<60mm)', 'long']
Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3

folder = 'BrainMapping'
# dur[0,:]       = np.int32(np.sum(abs(dur)))
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])

sub_path  ='X:\\4 e-Lab\\' # y:\\eLab
def update_sleep(subj, prot='BrainMapping', cond_folder='CR'):
    path_patient_analysis = sub_path+'\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    file_con = path_patient_analysis + '\\' + prot + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    con_trial = pd.read_csv(file_con)
    # load hypnogram
    file_hypno = path_patient_analysis + '\\stimlist_hypnogram.csv'  # path_patient + '/Analysis/stimlist_hypnogram.csv'
    if os.path.isfile(file_hypno):
        stimlist_hypno = pd.read_csv(file_hypno)
        stimlist_hypno.loc[(stimlist_hypno.sleep == 9), 'sleep'] = 0
        for ss in np.arange(5):
            stimNum = stimlist_hypno.loc[(stimlist_hypno.sleep == ss) & (stimlist_hypno.Prot == prot), 'StimNum']
            con_trial.loc[np.isin(con_trial.Num, stimNum), 'Sleep'] = ss

    con_trial.insert(5, 'SleepState', 'Wake')
    con_trial.loc[(con_trial.Sleep > 1) & (con_trial.Sleep < 4), 'SleepState'] = 'NREM'
    con_trial.loc[(con_trial.Sleep == 4), 'SleepState'] = 'REM'
    con_trial.loc[(con_trial.Sleep == 6), 'SleepState'] = 'SZ'
    con_trial.to_csv(file_con, index=False, header=True)  # return con_trial


def remove_art(con_trial, EEG_resp):
    # remove LL that are much higher than the mean
    # remove trials that have artefacts (high voltage values)

    chan, trial = np.where(np.max(abs(EEG_resp[0:int(0.5 * Fs)]), 2) > 1500)
    for i in range(len(trial)):
        con_trial.loc[
            (con_trial.Artefact == 0) & (con_trial.Chan == chan[i]) & (con_trial.Num == trial[i]), 'Artefact'] = -1

    resp_BL = abs(ff.lp_filter(EEG_resp, 2, Fs))
    resp_BL = resp_BL[:, :, 0:int(Fs)]
    resp_BL[resp_BL < 100] = 0
    AUC_BL = np.trapz(resp_BL, dx=1)
    chan, trial = np.where(AUC_BL > 60000)
    for i in range(len(trial)):
        con_trial.loc[
            (con_trial.Artefact == 0) & (con_trial.Chan == chan[i]) & (con_trial.Num == trial[i]), 'Artefact'] = -1

    # remove unrealistic high LL
    con_trial.loc[(con_trial.Artefact == 0) & (con_trial.LL > 30), 'Artefact'] = -1

    chan, trial = np.where(np.max(abs(EEG_resp), 2) > 4000)
    for i in range(len(trial)):
        con_trial.loc[
            (con_trial.Artefact == 0) & (con_trial.Chan == chan[i]) & (con_trial.Num == trial[i]), 'Artefact'] = 1

    return con_trial


########### Input
# for subj in ["EL004","EL005", "EL008", "EL010"]:  # "EL004","EL005","EL008",

def cal_con_trial(subj, cond_folder='Ph',skip_block=0, skip_single = 1):
    ######## General Infos
    print(subj + ' ---- START ------ ')

    # path_patient_analysis = 'Y:\\eLab\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_patient_analysis = sub_path+'\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj

    path_gen = os.path.join(sub_path+'\\Patients\\' + subj)
    if not os.path.exists(path_gen):
        path_gen = 'T:\\EL_experiment\\Patients\\' + subj
    path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    path_infos = os.path.join(path_gen, 'Electrodes')
    if not os.path.exists(os.path.join(path_infos, subj + "_labels.xlsx")):
        path_infos = os.path.join(path_gen, 'infos')
    if not os.path.exists(path_infos):
        path_infos = path_gen + '\\infos'

    sep = ';'
    Fs = 500
    Path(path_patient_analysis + '/' + folder + '/' + cond_folder + '/data').mkdir(parents=True, exist_ok=True)
    Path(path_patient_analysis + '/' + folder + '/' + cond_folder + '/figures/BM_plot_trial').mkdir(parents=True,
                                                                                                    exist_ok=True)
    Path(path_patient_analysis + '/' + folder + '/' + cond_folder + '/figures/BM_plot_trial_sig').mkdir(parents=True,
                                                                                                        exist_ok=True)
    Path(path_patient_analysis + '/' + folder + '/' + cond_folder + '/figures/BM_plot').mkdir(parents=True,
                                                                                              exist_ok=True)
    Path(path_patient_analysis + '/' + folder + '/' + cond_folder + '/surrogate').mkdir(parents=True,
                                                                                        exist_ok=True)

    # get labels
    if cond_folder == 'Ph':
        files_list = glob(path_patient_analysis + '/' + folder + '/data/Stim_list_*Ph*')
    else:
        files_list = glob(path_patient_analysis + '/' + folder + '/data/Stim_list_*')

    stimlist = pd.read_csv(files_list[
                               0])  # pd.read_csv(path_patient_analysis+'/' + folder + '/data/Stimlist.csv')# pd.read_csv(files_list[i])
    # EEG_resp = np.load(path_patient + '/Analysis/' + folder + '/data/ALL_resps_'+files_list[i][-11:-4]+'.npy')
    lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist,
        lbls)
    badchans = pd.read_csv(path_patient_analysis + '/BrainMapping/data/badchan.csv')
    bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])

    bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]
    # if condition is PHh, store concatenated file, since its only two blocks and not too large

    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'

    # file_MN1 = path_patient_analysis + '\\' + folder + '\\data\\M_N1.npy'
    ######### Load data
    rerun = 1  # todo: remove in future
    if rerun:  # rerun con_trial claculations / blockwise
        mx_across = 0
        for l in range(0, len(files_list)):  # for each file
            print('loading ' + files_list[l][-11:-4], end='\r')
            stimlist = pd.read_csv(files_list[l])
            if not ('noise' in stimlist.columns):
                stimlist.insert(9, 'noise', 0)

            new_col = ['StimNum', 'Num_block']
            for col in new_col:
                if col in stimlist:
                    stimlist = stimlist.drop(col, axis=1)
                stimlist.insert(4, col, np.arange(len(stimlist)))
            stimlist = stimlist.reset_index(drop=True)

            # con_trial_block = BMf.LL_BM_cond(EEG_resp, stimlist, 'h', bad_chans, coord_all, labels_clinic, StimChanSM, StimChanIx)
            block_l = files_list[l][-11:-4]
            file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_' + block_l + '.csv'
            if os.path.isfile(file) * skip_single:
                con_trial_block = pd.read_csv(file)
            else:
                EEG_resp = np.load(
                    path_patient_analysis + '/' + folder + '/data/ALL_resps_' + files_list[l][-11:-4] + '.npy')
                if EEG_resp.shape[1] != len(stimlist):
                    print('ERROR number of Stimulations is not correct')
                    print(EEG_resp.shape[1])
                    print(len(stimlist))
                    break
                else:

                    con_trial_block = BMf.LL_BM_connection(EEG_resp, stimlist, bad_chans, coord_all,
                                                           labels_clinic, StimChanSM, StimChanIx)
                    if cond_folder == 'CR':
                        con_trial_block = con_trial_block.drop(columns=['Condition'])
                    else:
                        con_trial_block = con_trial_block.drop(columns=['Sleep'])
                    con_trial_block = con_trial_block[~ np.isin(con_trial_block.Chan, bad_region)]
                    con_trial_block = con_trial_block[~ np.isin(con_trial_block.Stim, bad_region)]
                    con_trial_block = remove_art(con_trial_block, EEG_resp)
                    con_trial_block = con_trial_block.reset_index(drop=True)
                    con_trial_block.to_csv(file, index=False, header=True)
                # if (os.path.isfile(file_MN1)) & (
                #         'N1' not in con_trial_block):  # path_patient_analysis + '\\'+protocol+'\\data\\M_N1.npy', M_N1peaks :
                #     # todo: get p2p N1, N2, etc.
                #     M_N1peaks = np.load(file_MN1)
                #     con_trial_block = BMf.get_peaks_all(con_trial_block, EEG_resp, M_N1peaks)
                #     con_trial_block.to_csv(file, index=False, header=True)
            con_trial_block.Num = con_trial_block.Num_block + mx_across
            mx_across = mx_across + np.max(stimlist.StimNum) + 1  # np.max(con_trial_block.Num) + 1

            if l == 0:
                con_trial = con_trial_block
            else:
                con_trial = pd.concat([con_trial, con_trial_block])

        con_trial.to_csv(file_con, index=False, header=True)
    print(subj + ' ---- DONE ------ ')


def get_significance_trial(subj, cond_folder='CR', update_sig=0):
    ######## General Infos
    print(subj + ' ---- START ------ ')

    path_patient_analysis = sub_path+'\\EvM\\Projects\\EL_experiment\\Analysis\\Patients\\' + subj
    path_patient = sub_path+'\\Patients\\' + subj + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj

    sep = ';'
    Fs = 500

    stimlist = pd.read_csv(path_patient_analysis + '/' + folder + '/data/Stimlist.csv')  # pd.read_csv(files_list[i])
    lbls = pd.read_excel(path_patient + "/infos/" + subj + "_labels.xlsx", header=0, sheet_name='BP')
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist,
        lbls)
    badchans = pd.read_csv(path_patient_analysis + '/BrainMapping/data/badchan.csv')
    bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])
    labels_h = lbls.Hemisphere + '_' + labels_all
    bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]
    # only good channels that are stimulating
    non_stim = np.arange(len(labels_all))
    non_stim = np.delete(non_stim, StimChanIx, 0)
    WM_chans = np.where(labels_region == 'WM')[0]
    bad_all = np.unique(np.concatenate([WM_chans, bad_region, bad_chans, non_stim])).astype('int')
    stim_chans = np.arange(len(labels_all))
    stim_chans = np.delete(stim_chans, bad_all, 0)
    # get connection trial table
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    con_trial = pd.read_csv(file_con)

    if update_sig:
        file_thr = path_patient + '/Analysis/' + folder + '/' + cond_folder + '/data/threshold_blocks.npy'
        thr_blocks = np.load(file_thr)
        for sc in tqdm.tqdm(stim_chans, desc='Stimulation Channel'):
            for rc in range(len(labels_all)):
                con_trial, thr_blocks = sig_func.update_sig_val(sc, rc, thr_blocks, con_trial)
        con_trial.to_csv(file_con, index=False, header=True)
        # update con_trial and threhsold
        if (cond_folder == 'CR') & (np.max(con_trial.Sig) == 2):
            con_trial_1 = con_trial[(con_trial.Sig > -1) & (con_trial.Sig < 2)]
            con_trial_1 = con_trial_1.groupby(['Stim', 'Chan'])['Sig'].mean().reset_index(name='Prob')
            con_trial_1_1 = con_trial_1[con_trial_1.Prob > 0.2]
            # update con_trial list and threshold
            for i in range(len(con_trial_1_1)):
                sc = con_trial_1_1.Stim.values[i].astype('int')
                rc = con_trial_1_1.Chan.values[i].astype('int')
                con_trial.loc[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Sig == 1), 'Sig'] = 2
                thr_blocks[:, sc, rc, 2] = 1
            con_trial.loc[(con_trial.Sig == 1), 'Sig'] = 0
            con_trial.loc[(con_trial.Sig == 2), 'Sig'] = 1
            con_trial.to_csv(file_con, index=False, header=True)

            # np.save(file_thr, thr_blocks)
        print(subj + ' ----- Sig Updated ------ ')
        return

    print('loading EEG data')
    # file_MN1 = path_patient_analysis + '\\' + folder + '\\data\\M_N1.npy'
    # M_N1peaks = np.load(file_MN1)
    # M_N1peaks[M_N1peaks[:, :, 2] > 1, :] = 0
    EEG_resp_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy'
    EEG_CR = np.load(EEG_resp_CR_file)

    # Mean across all trials stored
    gt_file = path_patient_analysis + '\\' + folder + '/data/gt_data.npy'
    if os.path.isfile(gt_file):
        print('loading ground truth')
        gt_data = np.load(gt_file)
    else:
        gt_data = np.zeros((len(labels_all), len(labels_all), 2000))
        for sc in tqdm.tqdm(stim_chans, desc='Stimulation Channel'):
            for rc in range(len(labels_all)):  # tqdm.tqdm(stim_chans, desc='Response Channel', leave=False):
                gt_data = sig_func.get_gt_data(sc, rc, EEG_CR, con_trial, gt_data)
        np.save(gt_file, gt_data)
    repeat = 1
    # get pearson for each trial compared to ground truth
    if repeat:
        print('Calculating Pearson Coefficient for each trial')
        new_col = ['LL0', 'PLL', 'Sig']
        for col in new_col:
            if not col in con_trial: con_trial.insert(4, col, -1)
        for sc in tqdm.tqdm(stim_chans, desc='Stimulation Channel'):
            for rc in range(len(labels_all)):  # tqdm.tqdm(stim_chans, desc='Response Channel', leave=False):
                req = (con_trial.Artefact < 1) & (con_trial.Chan == rc) & (con_trial.Stim == sc)
                dat = con_trial[req]
                if len(dat) > 0:
                    con_trial = sig_func.get_trial_Pearson(sc, rc, con_trial, EEG_CR, w_p=0.1, w_LL=0.25, Fs=500, t_0=1,
                                                           t_resp=M_N1peaks[sc, rc, 2], p=abs(M_N1peaks[sc, rc, 1]),
                                                           gt_data=gt_data)

        con_trial.PLL = con_trial.LL0 * con_trial.Pearson
        con_trial.to_csv(file_con, index=False, header=True)
    # 3# get threshold value blockwise
    print('Calculating thresholds for each block')
    file_thr = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\threshold_blocks.npy'  # path_patient + '/Analysis/' + folder + '/' + cond_folder + '/data/threshold_blocks.npy'
    blocks = np.unique(con_trial.Block)
    # repeat = 0
    if os.path.isfile(file_thr):
        thr_blocks = np.load(file_thr)
        repeat = 0
    repeat = 1  # todo: remove
    if repeat:
        thr_blocks = np.zeros((len(blocks), len(labels_all), len(labels_all), 3))
        for sc in tqdm.tqdm(stim_chans, desc='Stimulation Channel'):
            for rc in range(
                    len(labels_all)):  # for rc in stim_chans:#tqdm.tqdm(stim_chans, desc='Response Channel', leave=True):
                thr_blocks, con_trial = sig_func.get_surr_connection(sc, rc, EEG_CR, thr_blocks, con_trial, w_LL=0.25,
                                                                     w_p=0.1,
                                                                     Fs=500, t_0=1, t_resp=0,
                                                                     p=abs(M_N1peaks[sc, rc, 1]), gt_data=gt_data)
        np.save(file_thr, thr_blocks)
        con_trial.to_csv(file_con, index=False, header=True)

    print(subj + ' ----- Sig Calculations  DONE ------ ')


def update_peaks(subj, cond_folder='CR'):
    ######## General Infos
    print(subj + ' ---- START ------ ')
    if platform.system() == 'Windows':
        sep = ','
        # path_patient_analysis = 'T:\EL_experiment\Projects\EL_experiment\Analysis\Patients\\' + subj
        path_patient_analysis = sub_path+'\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
        path_patient = 'T:\EL_experiment\Patients\\' + subj + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    else:  # 'Darwin' for MAC
        path_patient = '/Volumes/EvM_T7/PhD/EL_experiment/Patients/' + subj
        sep = ';'
    sep = ';'
    Fs = 500

    print('loading EEG data')
    EEG_resp_CR_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.npy'
    EEG_CR = np.load(EEG_resp_CR_file)

    file_MN1 = path_patient_analysis + '\\' + folder + '\\data\\M_N1.npy'
    M_N1peaks = np.load(file_MN1)
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    con_trial = pd.read_csv(file_con)

    con_trial = BMf.get_peaks_all(con_trial, EEG_CR, M_N1peaks)

    con_trial.to_csv(file_con, index=False, header=True)

    print(subj + ' ----- Sig Calculations  DONE ------ ')


#for subj in ['EL022']:  # 'EL004', 'EL005', 'EL008', 'EL010','EL012',,,,,, 'EL015','EL011', 'EL013',
    # if i>0: cal_con_trial(subj, 'CR')
    # _thread.start_new_thread(cal_con_trial, (subj, 'Ph')) # cal_con_trial(subj, 'Ph')
    ####old###get_significance_trial(subj, cond_folder='CR', update_sig=0)
    # cal_con_trial(subj, cond_folder='CR')
    # update_sleep(subj)
    # cal_con_trial(subj, cond_folder='Ph')

# for subj in ['EL016']:  # 'EL004', 'EL005', 'EL008', 'EL010','EL012',,,,,, 'EL015','EL011', 'EL013',
#     update_peaks(subj, cond_folder='CR')
# for subj in ['EL008','EL004']:  # 'EL004', 'EL005', 'EL008', 'EL010','EL012'
#     #if i>0: cal_con_trial(subj, 'CR')
#     cal_con_trial(subj, 'Ph')
#     get_significance_trial(subj, 'Ph')
#     #i = i+1
# #   _thread.start_new_thread(get_significance_trial, (subj, 'CR'))
# #while 1:
# #    time.sleep(1)
