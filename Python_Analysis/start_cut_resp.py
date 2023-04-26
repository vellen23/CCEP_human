import os
import numpy as np
import mne
import h5py
# import scipy.fftpack
import matplotlib
import pywt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# from scipy import signal
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

import LL_funcs
from pandas import read_excel
from scipy.stats import norm
import Ph_IO_analysis
import Ph_CR_analysis
from math import sin
import freq_funcs
import cut_resp
import glob
import _thread

cwd = os.getcwd()


def compute_cut(subj, skip_exist=1, prots = ['BM', 'IO', 'PP']):
    print(f'Performing calculations on {subj}')
    # path_patient = 'T:\EL_experiment\Patients\\' + subj  ##'T:\EL_experiment\Patients'#os.path.join('E:\PhD\EL_experiment\Patients',subj) # path_patient    = 'E:\PhD\EL_experiment\Patients\\'+subj ##'T:\EL_experiment\Patients'
    path_patient = 'Y:\\eLab\\Patients\\' + subj
    path_patient = 'X:\\4 e-Lab\\Patients\\'+ subj
    CUT = cut_resp.main(subj, path_patient)
    paths = os.listdir(path_patient + '\Data\EL_experiment')
    n_ex = len(paths)
    k = 0

    print('Cutting responses into [1,3]s epochs ... ')
    for n in range(n_ex):
        if not paths[n][:5] == 'infos':
            path_data = os.path.join(path_patient, 'Data\EL_experiment', paths[n], 'data_blocks')
            folders = glob.glob(path_data + '\\' + subj + '_*')
            if len(folders) > 0:
                for i in range(0, len(folders)): # todo: change back to 0
                    for prot in prots:

                        CUT.cut_resp(folders[i], k + i + 1, prot,skip_exist)

                k = k + i

    print(subj + ' ---- DONE ------ ')


def compute_list_update(subj, prots = ['BM', 'IO', 'PP']):
    print(f'Performing calculations on {subj}')
    path_patient = 'Y:\\eLab\\Patients\\' + subj  ##'T:\EL_experiment\Patients'#os.path.join('E:\PhD\EL_experiment\Patients',subj) # path_patient    = 'E:\PhD\EL_experiment\Patients\\'+subj ##'T:\EL_experiment\Patients'
    # path_patient    = '/Volumes/EvM_T7/PhD/EL_experiment/Patients/'+subj
    path_patient = 'X:\\4 e-Lab\\Patients\\' + subj
    CUT = cut_resp.main(subj, path_patient)
    paths = os.listdir(path_patient + '\Data\EL_experiment')
    n_ex = len(paths)
    n_ex = len(paths)
    k = 0  # todo: 0

    for n in range(n_ex):
        path_data = os.path.join(path_patient, 'Data\EL_experiment', paths[n], 'data_blocks')
        folders = glob.glob(path_data + '\\' + subj + '_*')

        if len(folders) > 0:
            for i in range(len(folders)):
                for prot in prots:
                    CUT.list_update(folders[i], k + i + 1, prot)

            k = k + i
    for prot in prots:
        CUT.concat_list(prot)
    print(subj + ' ---- DONE ------ ')


#
# compute_cut('EL014')
# for subj in ["EL022"]:  # , "EL010"
    # compute_list_update(subj)
    # compute_cut(subj, cut_blocks=1, concat_blocks=0)  # _thread.start_new_thread(compute_list_update, (subj,))
    # compute_cut(subj, concat_blocks=0)
    # compute_list_update(subj)
# while 1:
#    time.sleep(1)

