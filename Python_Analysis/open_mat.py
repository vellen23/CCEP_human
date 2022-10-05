import os
import numpy as np
import mne
import h5py
import pandas as pd
import sys
sys.path.append('./py_functions')
import hgp_func
from tkinter import *
import ntpath
from pandas import read_excel, ExcelWriter
root = Tk()
root.withdraw()
import copy
from scipy.io import savemat
import scipy
import platform
from glob import glob
#import SM_converter as SMC
cwd             = os.getcwd()

path_excel = 'Y:\\eLab\\Patients\\EL015\Data\\Clin_Stim\\'
path = 'Y:\\eLab\\Patients\\EL015\Data\\Clin_Stim\\data_blocks\\EL015_BP_ClinStim01'
lbls            = pd.read_excel(os.path.join(path_excel, "EL015_stimlist_ClinStim.xlsx"), header=0, sheet_name='channels')
labels_anat     = lbls.label_anat.values
labels_clinic   = lbls.label_BP.values
try:
    matfile         = h5py.File(path + "/ppEEG.mat", 'r')['ppEEG']
    EEGpp           = matfile[()].T
    Fs = h5py.File(path + "/ppEEG.mat", 'r')['fs']
except IOError:
    EEGpp           = scipy.io.loadmat(path + "/ppEEG.mat")['ppEEG']
    Fs = scipy.io.loadmat(path + "/ppEEG.mat")['fs']
    Fs = Fs[0][0]
TTL = pd.read_excel(os.path.join(path_excel, "EL015_stimlist_ClinStim.xlsx"), header=0, sheet_name='stim')
TTL = TTL.TTL_ds.values.astype('int')
inf                     = mne.create_info(ch_names=labels_anat.tolist()[:len(EEGpp)], sfreq=Fs, ch_types='seeg', verbose=None)
raw                     = mne.io.RawArray(EEGpp, inf, first_samp=0, copy='auto', verbose=None)
raw.info['lowpass']     = 200
raw.info['highpass']    = 0.5
