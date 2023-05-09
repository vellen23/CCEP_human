import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from glob import glob
import ntpath
import save_hypnogram
import start_cut_resp
import start_BM_blocks as BM_blocks
import start_IO_blocks as IO_blocks
import start_BM_CR as BM_CR
import concat
import run_sig_con
root = Tk()
root.withdraw()
sub_path  ='X:\\4 e-Lab\\' # y:\\eLab


### Preparation
subjs = ["EL024"]
for subj in subjs:
    ### cut data in epochs
    #start_cut_resp.compute_cut(subj, skip_exist=1, prots = ['BM']) # , 'IO'

    ### get con_trial
    BM_blocks.cal_con_trial(subj, cond_folder='CR',skip_block=1, skip_single = 1)
    # IO_blocks.cal_con_trial(subj, cond_folder='CR',skip_block=0, skip_single = 1)
    ### concat epoched data into h5py file, all responses accessible
    for f in ['BrainMapping']: # , 'InputOutput'
        concat.concat_resp_condition(subj, folder=f, cond_folder='CR', skip = 0)

###### BM Analysis
### get GT
run_sig_con.start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method = 'similarity', skipt_GT=1, skip_surr=1, trial_sig_labeling = 1)


