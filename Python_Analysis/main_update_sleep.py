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

subjs = ["EL022"]
for subj in subjs:
    # 1. read updated excel and update single csv
    # start_cut_resp.compute_list_update(subj=subj, prots = ['BM', 'IO']) #, 'PP'
    # 2. from single csv files update stimlist to updated stimlist_CR and updated con-trials
    save_hypnogram.run_main(subj, 1, 1, folders=['BrainMapping', 'InputOutput'])


print('Done')
