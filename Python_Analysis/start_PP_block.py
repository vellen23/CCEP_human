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
import analys_func
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
from matplotlib.patches import Rectangle
import tqdm
import similarity_funcs as sf
import BM_func as BMf
import IO_func as IOf
import PP_func as PPf
import NMF_funcs as NMFf
import freq_funcs as ff
from pathlib import Path
import _thread

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

# dur[0,:]       = np.int32(np.sum(abs(dur)))
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])

folder = 'PairedPulse'


def remove_art(con_trial, EEG_resp):
    # remove LL that are much higher than the mean
    # remove trials that have artefacts (high voltage values)
    chan, trial = np.where(np.max(abs(EEG_resp), 2) > 3000)
    for i in range(len(trial)):
        con_trial.loc[(con_trial.Chan == chan[i]) & (con_trial.Num_block == trial[i]), 'Artefact'] = -1

    chan, trial = np.where(np.max(abs(EEG_resp[0:int(0.5 * Fs)]), 2) > 1500)
    for i in range(len(trial)):
        con_trial.loc[(con_trial.Chan == chan[i]) & (con_trial.Num_block == trial[i]), 'Artefact'] = -1

    resp_BL = abs(ff.lp_filter(EEG_resp, 2, Fs))
    resp_BL = resp_BL[:, :, 0:int(Fs)]
    resp_BL[resp_BL < 100] = 0
    AUC_BL = np.trapz(resp_BL, dx=1)
    chan, trial = np.where(AUC_BL > 28000)
    for i in range(len(trial)):
        con_trial.loc[(con_trial.Chan == chan[i]) & (con_trial.Num_block == trial[i]), 'Artefact'] = -1

    # remove unrealistic high LL
    con_trial.loc[con_trial.LL > 40, 'Artefact'] = -1

    return con_trial


def compute_subj(subj):
    cond_folder = 'CR'  # Condition = 'Hour', 'Condition', 'Ph'

    if cond_folder == 'Ph':
        cond_vals = np.arange(4)
        cond_labels = ['BM', 'BL', 'Fuma', 'BZD']
        cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]
        cond1 = 'Condition'  # 'condition', 'h'
        cond_folder = 'Ph'  # 'Ph', 'Sleep', 'CR'
        Condition = 'Condition'
    if cond_folder == 'CR':
        Condition = 'Hour'  # Condition = 'Hour'
        cond1 = 'h'  # h (as stored in stimlist)

    ######## General Infos

    path_patient_analysis = 'y:\\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_gen = os.path.join('y:\\eLab\Patients\\' + subj)
    if not os.path.exists(path_gen):
        path_gen = 'T:\\EL_experiment\\Patients\\' + subj
    path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
    path_infos = os.path.join(path_patient, 'infos')
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
    rerun = 0
    if os.path.isfile(file_con):
        # con_trial
        con_trial = pd.read_csv(file_con)
    else:
        rerun = 1
    rerun = 1
    if rerun:
        mx_across = 0
        for l in range(0, len(files_list)):
            print('loading ' + files_list[l][-11:-4], end='\r')
            stimlist = pd.read_csv(files_list[l])
            if not ('noise' in stimlist.columns):
                stimlist.insert(9, 'noise', 0)
            if ('StimNum' in stimlist.columns):
                stimlist = stimlist.drop(columns='StimNum')
            stimlist.insert(5, 'StimNum', np.arange(len(stimlist)))

            EEG_resp = np.load(path_patient_analysis + '\\' +folder +'\\data\\ALL_resps_' + files_list[l][-11:-4] + '.npy')
            if EEG_resp.shape[1] != np.max(stimlist.StimNum) + 1:
                print('ERROR number of stimulations is not correct')
                break
            else:
                new_col = ['StimNum', 'Num_block']
                for col in new_col:
                    if col in stimlist:
                        stimlist = stimlist.drop(col, axis=1)
                    stimlist.insert(4, col, np.arange(len(stimlist)))
                stimlist = stimlist.reset_index(drop=True)
                # con_trial_block = BMf.LL_BM_cond(EEG_resp, stimlist, 'h', bad_chans, coord_all, labels_clinic, StimChanSM, StimChanIx)
                block_l = files_list[l][-11:-4]
                file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_' + block_l + '.csv'
                skip = 0
                if skip * os.path.isfile(file):
                    con_trial_block = pd.read_csv(file)
                else:
                    con_trial_block = PPf.get_LL_all_cond(EEG_resp, stimlist, lbls, bad_chans, w=0.25, Fs=500)
                    con_trial_block = remove_art(con_trial_block, EEG_resp)
                    con_trial_block = con_trial_block.reset_index(drop=True)
                    con_trial_block.to_csv(file, index=False, header=True)

                con_trial_block.Num = con_trial_block.Num_block + mx_across
                mx_across = mx_across + np.max(stimlist.StimNum) + 1  # np.max(con_trial_block.Num) + 1

                if l == 0:
                    con_trial = con_trial_block
                else:
                    con_trial = pd.concat([con_trial, con_trial_block])

        if ('zLL' in con_trial.columns):
            con_trial = con_trial.drop(columns='zLL')

        con_trial.insert(0, 'zLL', con_trial.groupby(['Stim', 'Chan', 'Int'])['LL'].transform(
            lambda x: (x - x.mean()) / x.std()).values)

        con_trial.loc[(con_trial.zLL > 6), 'Artefact'] = -1
        con_trial.loc[(con_trial.zLL < -5), 'Artefact'] = -1
        con_trial = con_trial.drop(columns='zLL')

        con_trial.to_csv(file_con, index=False, header=True)
        print(subj + ' ---- DONE ------ ')


########### Input
threads = 0

for subj in ["EL017"]: # ["EL005", "EL010", "EL016", "EL015","EL011", "EL004","El014"]:  # "EL004","EL005","EL008",EL004", "EL005", "EL008", "EL010
    if threads:
        _thread.start_new_thread(compute_subj(subj))
    else:
        print('start -- ' + subj)
        compute_subj(subj)
if threads:
    while 1:
        time.sleep(1)
