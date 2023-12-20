import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import seaborn as sns
import pandas as pd
import sys

sys.path.append('T:\EL_experiment\Codes\CCEP_human\Python_Analysis\py_functions')
from scipy.stats import norm
from tkinter import *
import _thread

root = Tk()
root.withdraw()
import scipy
import NMF_funcs as NMFf
import basic_func as bf
import BM_stats
from scipy import stats
from matplotlib.patches import Rectangle
import freq_funcs as ff
# from tqdm.notebook import trange, tqdm
# remove some warnings
import warnings
from pathlib import Path
import LL_funcs as LLf
import copy
import h5py
import BM_func as BMf
import graphNMF
import BM_plots
import graph_funcs
import load_summary as ls
import CCEP_senity
# I expect to see RuntimeWarnings in this block
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sleepstate_labels = ['NREM', 'REM', 'Wake']

folder = 'BrainMapping'
cond_folder = 'CR'
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])


class main:
    def __init__(self, subj):
        #  basics, get 4s of data for each stimulation, [-2,2]s
        self.folder = 'BrainMapping'
        self.cond_folder = 'CR'
        self.path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
        path_gen = os.path.join(sub_path + '\\Patients\\' + subj)
        if not os.path.exists(path_gen):
            path_gen = 'T:\\EL_experiment\\Patients\\' + subj
        path_patient = path_gen + '\Data\EL_experiment'  # os.path.dirname(os.path.dirname(cwd))+'/Patients/'+subj
        path_infos = os.path.join(path_gen, 'Electrodes')
        if not os.path.exists(os.path.join(path_infos, subj + "_labels.xlsx")):
            path_infos = os.path.join(path_gen, 'infos')
        if not os.path.exists(path_infos):
            path_infos = path_gen + '\\infos'

        self.Fs = 500
        self.dur = np.zeros((1, 2), dtype=np.int32)
        self.dur[0, :] = [-1, 3]
        self.dur_tot = np.int32(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        # load patient specific information
        lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
        if "type" in lbls:
            lbls = lbls[lbls.type == 'SEEG']
            lbls = lbls.reset_index(drop=True)
        self.labels_all = lbls.label.values
        self.labels_C = lbls.Clinic.values
        self.hemisphere = lbls.Hemisphere
        stimlist = pd.read_csv(
            self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\stimlist_' + self.cond_folder + '.csv')
        if len(stimlist) == 0:
            stimlist = pd.read_csv(
                self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\stimlist_' + self.cond_folder + '.csv')
        #
        labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
            stimlist,
            lbls)
        bad_region = np.where((labels_region == 'Unknown') | (labels_region == 'WM') | (labels_region == 'OUT') | (
                labels_region == 'Putamen'))[0]
        self.labels_region_L = lbls.Hemisphere.values + '_' + labels_region
        self.subj = subj
        self.lbls = lbls
        atlas_regions = pd.read_excel(
            "X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients\\Across\\elab_labels.xlsx",
            sheet_name="atlas")
        self.labels_region = labels_region
        for i in range(len(labels_all)):
            area_sel = " ".join(re.findall("[a-zA-Z_]+", labels_all[i]))
            self.labels_region[i] = atlas_regions.loc[atlas_regions.Abbreviation == area_sel, "Region"].values[0]
        # self.labels_region = labels_region

        # regions information
        # self.CR_color = pd.read_excel("T:\EL_experiment\Patients\\" + 'all' + "\Analysis\BrainMapping\CR_color.xlsx",
        #                               header=0)
        regions = pd.read_excel(sub_path + "\\EvM\Projects\EL_experiment\Analysis\Patients\Across\elab_labels.xlsx",
                                sheet_name='regions',
                                header=0)
        self.color_regions = regions.color.values
        self.regions = regions
        badchans = pd.read_csv(self.path_patient_analysis + '/BrainMapping/data/badchan.csv')
        self.bad_chans = np.unique(np.array(np.where(badchans.values[:, 1:] == 1))[0, :])
        # C = regions.label.values
        # self.path_patient   = path_patient
        # self.path_patient_analysis = os.path.join(os.path.dirname(os.path.dirname(self.path_patient)), 'Projects\EL_experiment\Analysis\Patients', subj)
        ##bad channels
        non_stim = np.arange(len(self.labels_all))
        StimChanIx = np.unique(np.array(StimChanIx).astype('int'))
        non_stim = np.delete(non_stim, StimChanIx, 0)
        WM_chans = np.where(self.labels_region == 'WM')[0]
        self.bad_all = np.unique(np.concatenate([WM_chans, bad_region, self.bad_chans, non_stim])).astype('int')
        stim_chans = np.arange(len(labels_all))
        self.stim_chans = np.delete(stim_chans, self.bad_all, 0)
        Path(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures').mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\NNMF').mkdir(
            parents=True, exist_ok=True)
        Path(self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\NNMF\\figures').mkdir(
            parents=True, exist_ok=True)
        for group in ['\\General', '\\Block', '\\Sleep']:
            for metric in ['\\BM_LL', '\\BM_Prob', '\\BM_binary', '\\BM_sym', '\\BM_change', '\\BM_Dir']:
                Path(
                    self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\BM_figures' + group + metric).mkdir(
                    parents=True, exist_ok=True)

        # labels:
        labels_sel = np.delete(self.labels_all, self.bad_all, 0)
        areas_sel = np.delete(self.labels_region_L, self.bad_all, 0)
        # sort
        ind = np.argsort(areas_sel)
        areas_sel = np.delete(self.labels_region, self.bad_all, 0)
        self.labels_sel = labels_sel[ind]
        self.areas_sel = areas_sel[ind]
        self.ind = ind

    def get_SNR(self, con_trial, EEG_resp, skip=1):
        con_trial = bf.add_sleepstate(con_trial)
        # for ss in ['Wake']:
        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\SNR_all.csv'  # summary_general
        if os.path.isfile(summary_gen_path) * skip:
            print('SNR  already calculated  -  skipping .. ')
        else:
            start = 1
            for sc in range(len(self.labels_all)):
                for rc in range(len(self.labels_all)):
                    tab = CCEP_senity.calculate_snr(sc, rc, con_trial,EEG_resp, win_LL=0.25, win_p2p=0.5, t0=1, Fs=500)
                    tab = pd.DataFrame(tab, columns=['Stim', 'Chan', 'Sig', 'N', 'Real', 'P2P', 'LL', 'P2P_SNR',
                                                     'LL_SNR'])
                    if len(tab)>0:
                        if start:
                            table = tab
                            start = 0
                        else:
                            table = pd.concat([table, tab], 0)
            table = table.reset_index(drop=True)
            table.to_csv(summary_gen_path, index=False, header=True)  # get_con_summary_wake


def start_subj(subj, cluster_method='similarity'):
    print(subj + ' -- START --')
    run_main = main(subj)
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    # load data
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
     # todo: make clean
    con_trial = pd.read_csv(file_con)

    h5_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.h5'
    print('loading h5')
    EEG_resp = h5py.File(h5_file)
    EEG_resp = EEG_resp['EEG_resp']

    run_main.get_SNR(con_trial, EEG_resp, skip=0)
    print(subj + ' ----- DONE')


thread = 0
sig = 0
subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL019", "EL020", "EL021",
         "EL022", "EL024", "EL026", "EL027", "EL028"]

for subj in subjs:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'"EL021", "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016",
    if thread:
        _thread.start_new_thread(start_subj, (subj, sig))
    else:
        start_subj(subj, 'similarity')
if thread:
    while 1:
        time.sleep(1)

print('Done')
