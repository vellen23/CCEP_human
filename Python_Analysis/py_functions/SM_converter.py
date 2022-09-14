import numpy as np
import pandas as pd
import os
import sys
import xlrd
from pandas import read_excel, ExcelWriter


def get_converter(subj):
    path_patient    = '/Volumes/EvM_T7/PhD/EL_experiment/Patients/'+subj+'/infos'
    # path               = os.path.dirname(os.path.abspath(name)) #current path
    # path_patient       = os.path.join(path, 'Patients', name) # '.../Insel/NELS/Source/Patients/P001'

    file_name           = subj + '_lookup.xlsx'  # name of your excel file
    path_file           = os.path.join(path_patient, file_name)
    df                  = read_excel(path_file, sheet_name='SM2_conv')
    SM2                 = df['SM2'].values
    SM1                 = df['chan_num'].values
    conv = np.stack([SM1, SM2]).T

    return conv


