import os
import numpy as np
import time
import pandas as pd
import sys
import glob
sys.path.append('T:\EL_experiment\Codes\CCEP_human\Python_Analysis\py_functions')
from scipy.stats import norm
from tkinter import *
import _thread

root = Tk()
root.withdraw()
import scipy

import basic_func as bf
from matplotlib.patches import Rectangle
# from tqdm.notebook import trange, tqdm
# remove some warnings
import warnings
from pathlib import Path
import LL_funcs as LLf
import copy
import h5py
import IO_func as IOf
import IO_func_across as IOf_across
import load_summary as ls

# I expect to see RuntimeWarnings in this block
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sleepstate_labels = ['NREM', 'REM', 'Wake']

folder = 'InputOutput'
cond_folder = 'CR'
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])


class main:
    def __init__(self, subj):
        #  basics, get 4s of data for each stimulation, [-2,2]s
        self.folder = 'InputOutput'
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
        self.coord_all= coord_all
        atlas_regions = pd.read_excel(
            "X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients\\Across\\elab_labels.xlsx",
            sheet_name="atlas")
        self.labels_region = labels_region
        for i in range(len(labels_all)):
            area_sel = " ".join(re.findall("[a-zA-Z_]+", labels_all[i]))
            # print(area_sel)
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
        WM_chans = np.where(self.labels_region == 'WM')[0]
        self.bad_all = np.unique(np.concatenate([WM_chans, bad_region, self.bad_chans])).astype('int')
        stim_chans = np.arange(len(labels_all))
        self.stim_chans = np.delete(stim_chans, self.bad_all, 0)

    def get_AUC_mean(self, con_trial, EEG_resp, skip):
        con_trial = con_trial[~np.isin(con_trial.Chan, self.bad_all)].reset_index(drop=True)
        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\AUC_mean.csv'  # summary_general
        if os.path.isfile(summary_gen_path) * skip:
            print('AUC-mean already calculated  -  skipping .. ')
        else:
            df = IOf_across.save_AUC_connection(con_trial, EEG_resp)
            df = ls.adding_distance(df, self.coord_all)
            df = ls.adding_area(df, self.lbls, pair=1)
            df = ls.adding_region(df, pair=1)
            df = ls.adding_subregion(df, pair=1)
            df.to_csv(summary_gen_path, index=False, header=True)  # get_con_summary_wake

    def add_delay(self, con_trial, EEG_resp):
        con_trial = con_trial[
            ~np.isin(con_trial.Chan, self.bad_all)].reset_index(
            drop=True)

        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\AUC_mean.csv'  # summary_general
        auc_summary = pd.read_csv(summary_gen_path)
        auc_summary = IOf_across.get_delay(con_trial, EEG_resp, auc_summary)
        auc_summary.to_csv(summary_gen_path, index=False, header=True)  # get_con_summary_wake
    def get_AUC_sleepstate(self, con_trial, EEG_resp, skip = 1):
        SleepStates_val = ['Wake', 'NREM', 'REM']
        con_trial = con_trial[
            ~np.isin(con_trial.Chan, self.bad_all) & np.isin(con_trial.SleepState, SleepStates_val)].reset_index(
            drop=True)

        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\AUC_mean.csv'  # summary_general
        auc_summary = pd.read_csv(summary_gen_path)
        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\AUC_mean_SS.csv'  # summary_general
        if os.path.isfile(summary_gen_path) * skip:
            print('AUC-mean already calculated  -  skipping .. ')
        else:
            df = IOf_across.get_AUC_stats_across_connections(con_trial, auc_summary, EEG_resp, SleepStates_val)
            df = ls.adding_distance(df, self.coord_all)
            df = ls.adding_area(df, self.lbls, pair=1)
            df = ls.adding_region(df, pair=1)
            df = ls.adding_subregion(df, pair=1)
            df.to_csv(summary_gen_path, index=False, header=True)  # get_con_summary_wake

    def get_AUC_sleepstate_subnetwork(self, con_trial):
        summary_gen_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\data\\NMF_summary.csv'

        # Initialize an empty list to store DataFrames
        dfs = []

        for sc in np.unique(con_trial.Stim).astype('int'):
            nmf_path = self.path_patient_analysis + '\\' + self.folder + '\\' + self.cond_folder + '\\NNMF_Stim\\' + self.labels_all[sc] + '/'

            # Find files that match the pattern
            files = glob.glob(nmf_path + 'IO_SleepState_AUC_1rk*.csv')

            # Check if any files were found before attempting to read
            if files:
                df_single = pd.read_csv(files[0])  # Reads the first file matching the pattern
                dfs.append(df_single)  # Append the DataFrame to the list

        # Concatenate all DataFrames in the list, if it is not empty
        if dfs:
            df = pd.concat(dfs, ignore_index=True)  # Use ignore_index=True to reindex the new DataFrame

            # Assuming `ls.adding_area`, `ls.adding_region`, `ls.adding_subregion` are methods that modify the DataFrame
            # and `self.lbls`, `pair` are parameters for these methods
            df['Chan'] = df['Stim']
            df = ls.adding_area(df, self.lbls, pair=0)
            df = ls.adding_region(df, pair=0)
            df = ls.adding_subregion(df, pair=0)

            # Save the concatenated DataFrame to a CSV file
            df.to_csv(summary_gen_path, index=False, header=True)
        else:
            print("No CSV files were found matching the pattern.")

def start_subj(subj, cluster_method='similarity'):
    print(subj + ' -- START --')
    run_main = main(subj)
    path_patient_analysis = sub_path + '\\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    # load data
    file_con = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    # todo: make clean
    con_trial = pd.read_csv(file_con)
    con_trial = bf.add_sleepstate(con_trial)
    con_trial['Num'] = con_trial['Num'].astype('int')
    h5_file = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_' + cond_folder + '.h5'
    print('loading h5')
    EEG_resp = h5py.File(h5_file)
    EEG_resp = EEG_resp['EEG_resp']
    run_main.get_AUC_sleepstate_subnetwork(con_trial)
    #run_main.get_AUC_mean( con_trial, EEG_resp, 1)
    # add delay
    #run_main.add_delay(con_trial, EEG_resp)
    #run_main.get_AUC_sleepstate(con_trial, EEG_resp, 1)
    print(subj + ' ----- DONE')


thread = 0
sig = 0

subjs = ["EL015", "EL016", "EL019", "EL020", "EL021",
         "EL022", "EL024", "EL026", "EL027", "EL028","EL004","EL005", "EL014","EL010", "EL011", "EL012", "EL013"]
subjs = ["EL004","EL005","EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL019", "EL021",
         "EL022", "EL024", "EL026", "EL027", "EL020", "EL028"]
for subj in subjs:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'"EL021", "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016",
    if thread:
        _thread.start_new_thread(start_subj, (subj, sig))
    else:
        start_subj(subj, 'similarity')
if thread:
    while 1:
        time.sleep(1)

print('Done')
