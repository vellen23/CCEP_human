import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from glob import glob
import ntpath
import save_hypnogram
import start_cut_resp

root = Tk()
root.withdraw()
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab

subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL019", "EL020", "EL021", "EL022",
         "EL024", "EL026", "EL027", "EL028"]
# subjs = ["EL026", "EL027", "EL028", "EL022"]

subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL019", "EL020", "EL021", "EL022",
         "EL024", "EL026", "EL027", "EL028"]


subjs = [ "EL019", "EL020", "EL021", "EL022",
         "EL024"]

for subj in subjs:
    # 1. read updated excel and update single csv
    start_cut_resp.compute_list_update(subj=subj, prots = ['BM', 'IO', 'PP']) #, 'PP'
    # 2. from single csv files update stimlist to updated stimlist_CR and updated con-trials
    save_hypnogram.run_main(subj, update_list=1, update_contrial=0, folders=['BrainMapping', 'InputOutput', 'PairedPulse'])

print('Done')
# done EL010, EL011, EL012, EL013, EL014, EL015