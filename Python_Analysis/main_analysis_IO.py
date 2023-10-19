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
# import start_BM_CR as BM_CR
import concat
import run_sig_con
import save_hypnogram

root = Tk()
root.withdraw()
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def run_first(subj, cut=1, con_trial=1, hyp=0, concat=1):
    ### cut data in epochs
    if cut:
        start_cut_resp.compute_cut(subj, skip_exist=0, prots=['BM', 'IO'])  # , 'IO'

    ### get con_trial
    if con_trial:
        if 'BM' in prots:
            BM_blocks.cal_con_trial(subj, cond_folder='CR', skip_block=0, skip_single=1)
        if 'IO' in prots:
            IO_blocks.cal_con_trial(subj, cond_folder='CR', skip_block=0, skip_single=1)
        if 'PP' in prots:
            PP_blocks.cal_con_trial(subj, cond_folder='CR', skip_block=0, skip_single=1)
    if hyp:
        save_hypnogram.run_main(subj, 1, 1, folders=['BrainMapping', 'InputOutput'])
    ### concat epoched data into h5py file, all responses accessible
    if concat:
        for p in prots:  # , 'InputOutput'
            if p == 'BM':
                f = 'BrainMapping'
            elif p == 'IO':
                f = 'InputOutput'
            else:
                f = 'PairedPulse'
            concat.concat_resp_condition(subj, folder=f, cond_folder='CR', skip=1)

    ###### BM Analysis
    ### get GT
    run_sig_con.start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method='similarity', skipt_GT=1,
                              skip_surr=1, trial_sig_labeling=1)


def run_update(subj):
    ### get con_trial
    # todo: ATTENTION deletes Significance testing !!
    # BM_blocks.update_timestamp(subj, cond_folder='CR')
    # IO_blocks.update_timestamp(subj, cond_folder='CR')
    ### get GT
    # BM_blocks.cal_con_trial(subj, cond_folder='CR', skip_block=0, skip_single=0)
    run_sig_con.start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method='similarity', skipt_GT=0,
                              skip_surr=1, trial_sig_labeling=1)
    # IO_blocks.cal_con_trial(subj, cond_folder='CR', skip_block=0, skip_single=1)
    # ### concat epoched data into h5py file, all responses accessible
    # for f in ['BrainMapping', 'InputOutput']:  # , 'InputOutput'
    #     concat.concat_resp_condition(subj, folder=f, cond_folder='CR', skip=1)

    ###### BM Analysis
    ### get GT
    # run_sig_con.start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method='similarity', skipt_GT=1,
    #                          skip_surr=1, trial_sig_labeling=1)


def run_update_BM(subj):
    # BM_blocks.cal_con_trial(subj, cond_folder='CR', skip_block=0, skip_single=0)
    run_sig_con.start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method='similarity',
                              skipt_GT=1,
                              skip_surr=1, trial_sig_labeling=1)


subjs = ["EL026"]
for subj in subjs:
    run_update_BM(subj)
    # run_first(subj, cut = 1, con_trial=1, hyp = 0)
print('Done')
