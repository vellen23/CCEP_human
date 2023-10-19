import os
import numpy as np
import mne
import imageio
import h5py
import matplotlib
import pywt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import seaborn as sns
import pandas as pd
import matplotlib.mlab as mlab
import sys

sys.path.append('./py_functions')
from scipy.stats import norm
import LL_funcs
from scipy.stats import norm
from tkinter import filedialog
from tkinter import *
import ntpath

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
import PP_func as PPf
import freq_funcs as ff
from pathlib import Path
import _thread

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

dist_groups = np.array([[0, 30], [30, 60], [60, 120]])
dist_labels = ['local (<30 mm)', 'short (<60mm)', 'long']
Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3

# dur[0,:]       = np.int32(np.sum(abs(dur)))
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])

folder = 'PairedPulse'
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def remove_art(con_trial, EEG_resp):
    # remove LL that are much higher than the mean
    # remove trials that have artefacts (high voltage values)
    chan, trial = np.where(np.max(abs(EEG_resp), 2) > 3000)
    for i in range(len(trial)):
        con_trial.loc[(con_trial.Artefact == 0) & (con_trial.Chan == chan[i]) & (
                con_trial.Num_block == trial[i]), 'Artefact'] = -1

    chan, trial = np.where(np.max(abs(EEG_resp[0:int(0.5 * Fs)]), 2) > 1500)
    for i in range(len(trial)):
        con_trial.loc[(con_trial.Artefact == 0) & (con_trial.Chan == chan[i]) & (
                con_trial.Num_block == trial[i]), 'Artefact'] = -1

    resp_BL = abs(ff.lp_filter(EEG_resp, 2, Fs))
    resp_BL = resp_BL[:, :, 0:int(Fs)]
    resp_BL[resp_BL < 100] = 0
    AUC_BL = np.trapz(resp_BL, dx=1)
    chan, trial = np.where(AUC_BL > 28000)
    for i in range(len(trial)):
        con_trial.loc[(con_trial.Artefact == 0) & (con_trial.Chan == chan[i]) & (
                con_trial.Num_block == trial[i]), 'Artefact'] = -1

    # remove unrealistic high LL
    con_trial.loc[(con_trial.Artefact == 0) & (con_trial.LL > 40), 'Artefact'] = -1

    return con_trial


def cal_con_trial(subj, cond_folder='CR', skip_block=1, skip_single=1):
    # cond_folder = 'CR'  # Condition = 'Hour', 'Condition', 'Ph'
    print(f'Performing PP calculations on {subj}, Condition: ' + cond_folder)

    path_patient_analysis = sub_path + '\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_gen = os.path.join(sub_path + '\Patients\\' + subj)
    if not os.path.exists(path_gen):
        path_gen = 'T:\\EL_experiment\\Patients\\' + subj
    path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    path_infos = os.path.join(path_gen, 'Electrodes')
    if not os.path.exists(os.path.join(path_infos, subj + "_labels.xlsx")):
        path_infos = os.path.join(path_gen, 'infos')
    if not os.path.exists(path_infos):
        path_infos = path_gen + '\\infos'
    Fs = 500
    Path(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\').mkdir(parents=True, exist_ok=True)
    Path(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\').mkdir(parents=True, exist_ok=True)

    # get labels
    if cond_folder == 'Ph':
        files_list = glob(path_patient_analysis + '\\' + folder + '\\data/Stim_list_*Ph*')
    elif cond_folder == 'CR':
        files_list = glob(path_patient_analysis + '\\' + folder + '\\data/Stim_list_*CR*')
    else:
        files_list = glob(path_patient_analysis + '\\' + folder + '\\data/Stim_list_*')
    i = 0
    stimlist = pd.read_csv(files_list[i])
    # EEG_resp = np.load(path_patient + '/Analysis/' + folder + '/data/ALL_resps_'+files_list[i][-11:-4]+'.npy')
    lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist,
        lbls)
    badchans = pd.read_csv(path_patient_analysis + '/BrainMapping/data/badchan.csv')
    bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])
    # bad_region = np.where((labels_region == 'WM') | (labels_region == 'OUT') | (labels_region == 'Putamen'))[0]
    # file_con = path_patient + '/Analysis/' + folder + '/' + cond_folder + '/data/con_trial_all.csv'
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'

    ######### Load data
    if os.path.isfile(file_con) * skip_block:
        # con_trial
        con_trial = pd.read_csv(file_con)
        rerun = 0
    else:
        rerun = 1
    # file_MN1 = path_patient_analysis + '\\' + folder + '\\data\\M_N1.npy'
    if rerun:
        mx_across = 0
        for l in range(0, len(files_list)):
            print('loading ' + files_list[l][-11:-4], end='\r')
            stimlist = pd.read_csv(files_list[l])
            if not ('noise' in stimlist.columns):
                stimlist.insert(9, 'noise', 0)
            stimlist['Time'] = pd.to_datetime(stimlist['date'].astype('int'), format='%Y%m%d') + pd.to_timedelta(
                stimlist['h'], unit='H') + \
                               pd.to_timedelta(stimlist['min'], unit='Min') + \
                               pd.to_timedelta(stimlist['s'], unit='Sec')
            new_col = ['StimNum', 'Num_block']
            for col in new_col:
                if col in stimlist:
                    stimlist = stimlist.drop(col, axis=1)
                stimlist.insert(4, col, np.arange(len(stimlist)))
            stimlist = stimlist.reset_index(drop=True)
            stimlist['Num_block'] = stimlist['StimNum']
            # con_trial_block = BMf.LL_BM_cond(EEG_resp, stimlist, 'h', bad_chans, coord_all, labels_clinic, StimChanSM, StimChanIx)
            block_l = files_list[l][-11:-4]
            file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_' + block_l + '.csv'
            if os.path.isfile(file) * skip_single:
                con_trial_block = pd.read_csv(file)
            else:
                EEG_resp = np.load(
                    path_patient_analysis + '\\' + folder + '\\data\\ALL_resps_' + files_list[l][-11:-4] + '.npy')
                if EEG_resp.shape[1] != np.max(stimlist.StimNum) + 1:
                    print('ERROR number of stimulations is not correct')
                    break
                else:
                    con_trial_block = PPf.get_LL_all_cond(EEG_resp, stimlist, lbls, bad_chans, w=0.25, Fs=Fs)
                    con_trial_block = con_trial_block.merge(stimlist[['Num_block', 'Time']], on='Num_block')
                    con_trial_block = remove_art(con_trial_block, EEG_resp)
                    con_trial_block = con_trial_block.reset_index(drop=True)
                    con_trial_block.Chan = con_trial_block.Chan.astype('int')
                    con_trial_block.Stim = con_trial_block.Stim.astype('int')
                    con_trial_block.Num = con_trial_block.Num.astype('int')
                    con_trial_block.Num_block = con_trial_block.Num_block.astype('int')
                    con_trial_block.to_csv(file, index=False, header=True)
            con_trial_block.Num = con_trial_block.Num_block + mx_across
            mx_across = mx_across + np.max(stimlist.StimNum) + 1  # np.max(con_trial_block.Num) + 1
            if l == 0:
                con_trial = con_trial_block
            else:
                con_trial = pd.concat([con_trial, con_trial_block])

        # if ('zLL' in con_trial.columns):
        #     con_trial = con_trial.drop(columns='zLL')
        #
        # con_trial.insert(0, 'zLL', con_trial.groupby(['Stim', 'Chan', 'IPI', 'Intc'])['LL'].transform(
        #     lambda x: (x - x.mean()) / x.std()).values)
        #
        # con_trial.loc[(con_trial.Artefact == 0) &(con_trial.zLL > 6), 'Artefact'] = -1
        # con_trial.loc[(con_trial.Artefact == 0) &(con_trial.zLL < -5), 'Artefact'] = -1
        # con_trial = con_trial.drop(columns='zLL')

        con_trial.to_csv(file_con, index=False, header=True)
        print(subj + ' ---- DONE ------ ')
