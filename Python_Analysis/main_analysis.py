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
import start_PP_block as PP_blocks
# import start_BM_CR as BM_CR
import time
import concat
import run_sig_con

root = Tk()
root.withdraw()
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

def run_keller(subj):
    run_sig_con.sig_con_keller(subj)
    print(subj + ' ---- DONE')
def update(subj):
    run_sig_con.start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method='similarity', skipt_GT=1,
                              skip_surr=1, trial_sig_labeling=1)
    print(subj + ' ---- DONE')
def run_first(subj, prots=['BM', 'IO', 'PP'], cut=1, con_trial=1, hyp=0, run_concat=1):
    ### cut data in epochs
    if cut:
        start_cut_resp.compute_cut(subj, skip_exist=1, prots=['BM', 'IO'])  # , 'IO'

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
    if run_concat:
        for p in prots:  # , 'InputOutput'
            if p == 'BM':
                f = 'BrainMapping'
            elif p == 'IO':
                f = 'InputOutput'
            else:
                f = 'PairedPulse'
            concat.concat_resp_condition(subj, folder=f, cond_folder='CR', skip=0)

    ###### BM Analysis
    ### get GT

    run_sig_con.start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method='similarity', skipt_GT=1,
                              skip_surr=1, trial_sig_labeling=1)

def run_update(subj):

    #BM_blocks.clean_contrial(subj, cond_folder='CR')
    # run_sig_con.start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method='similarity',
    #                           skipt_GT=1,
    #                           skip_surr=1, skip_summ=1, trial_sig_labeling=1)
    run_sig_con.trial_significance(subj, folder='BrainMapping', cond_folder='CR')

subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL017", "EL019", "EL020", "EL021",
         "EL022", "EL024", "EL025", "EL026", "EL027",'EL028']
thread = 0
for subj in subjs:  # ''El009', 'EL010', 'EL011', 'EL012', 'EL013', 'EL015', 'EL014','EL016', 'EL017'"EL021", "EL010", "EL011", "EL012", 'EL013', 'EL014', "EL015", "EL016",
    if thread:
        import _thread
        _thread.start_new_thread(update, (subj, ))
    else:
        run_update(subj)
if thread:
    while 1:
        time.sleep(1)