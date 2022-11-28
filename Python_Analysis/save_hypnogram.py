import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from glob import glob
import ntpath
root = Tk()
root.withdraw()

def update_sleep(subj, prot='BrainMapping', cond_folder = 'CR'):
    path_patient_analysis = 'y:\\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    file_con = path_patient_analysis + '\\' + prot + '\\' + cond_folder + '\\data\\con_trial_all.csv'
    con_trial =pd.read_csv(file_con)
    # load hypnogram
    file_hypno  = path_patient_analysis+'\\stimlist_hypnogram.csv' #path_patient + '/Analysis/stimlist_hypnogram.csv'
    if os.path.isfile(file_hypno):
        stimlist_hypno = pd.read_csv(file_hypno)
        stimlist_hypno.loc[(stimlist_hypno.sleep == 9),'sleep']=0
        for ss in np.arange(5):
            stimNum = stimlist_hypno.loc[(stimlist_hypno.sleep == ss) & (stimlist_hypno.Prot == prot), 'StimNum']
            con_trial.loc[np.isin(con_trial.Num, stimNum), 'Sleep'] = ss

    if 'SleepState' not in con_trial:
        con_trial.insert(5, 'SleepState', 'Wake')
    con_trial.loc[(con_trial.Sleep > 0) & (con_trial.Sleep < 4), 'SleepState'] = 'NREM'
    con_trial.loc[(con_trial.Sleep == 4), 'SleepState'] = 'REM'
    con_trial.loc[(con_trial.Sleep == 6), 'SleepState'] = 'SZ'
    con_trial.to_csv(file_con, index=False, header=True) # return con_trial

def update_stimlist(subj, folder='InputOutput', cond_folder='CR'):
    # is start_cut_resp updated?
    path_patient_analysis = 'y:\\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    files = glob(path_patient_analysis + '\\' + folder + '\\data\\Stim_list_*' + cond_folder + '*')
    files = np.sort(files)
    # prots           = np.int64(np.arange(1, len(files) + 1))  # 43
    stimlist = []
    EEG_resp = []
    conds = np.empty((len(files),), dtype=object)
    for p in range(len(files)):
        file = files[p]
        # file = glob(self.path_patient + '/Analysis/'+folder+'/data/Stim_list_' + str(p) + '_*')[0]
        idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
        cond = ntpath.basename(file)[idxs[-2] - 2:idxs[-2]]  # ntpath.basename(file)[idxs[-2] + 2:-4]  #
        conds[p] = cond
        print(str(p + 1) + '/' + str(len(files)) + ' -- All_resps_' + file[-11:-4])
        stim_table = pd.read_csv(file)
        stim_table['type'] = cond
        if len(stimlist) == 0:
            stimlist = stim_table
        else:
            stimlist = pd.concat([stimlist, stim_table])
    stimlist = stimlist.drop(columns="StimNum", errors='ignore')
    stimlist = stimlist.fillna(0)
    stimlist = stimlist.reset_index(drop=True)
    col_drop = ["StimNum", 'StimNum.1', 's', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
    for d in range(len(col_drop)):
        if (col_drop[d] in stimlist.columns):
            stimlist = stimlist.drop(columns=col_drop[d])
    stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)
    stimlist.to_csv(
        path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\stimlist_' + cond_folder + '.csv',
        index=False,
        header=True)  # scat_plot
    print('data stored')

subj            = "EL012"
update = 1

#subjs = ['EL003', 'EL004', 'EL005', 'EL010', 'EL011']
subjs = ["EL004", "EL005"]
for subj in subjs:
    cwd             = os.getcwd()
    print(subj)
    if update:
        for f in ['InputOutput','PairedPulse']:  # 'BrainMapping',
            update_stimlist(subj, folder=f, cond_folder='CR')

    path_patient = 'Y:\eLab\Patients\\'+subj
    sep =';'

    file_hypno = 'Y:\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj+ '\\stimlist_hypnogram.csv'
    #file_hypno_all = 'C:\\Users\i0328442\Desktop\hypnograms\\'+subj+'_stimlist_hypnogram.csv'
    file_hypno_fig = 'Y:\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj+ '\\stimlist_hypnogram.svg'
    color_elab      = np.zeros((4,3))
    color_elab[0,:] = np.array([31, 78, 121])/255
    color_elab[1,:] = np.array([189, 215, 238])/255
    color_elab[2,:] = np.array([0.256, 0.574, 0.431])
    color_elab[3,:] = np.array([1, 0.574, 0])


    stimlist_hypno =[]
    prots = ['BrainMapping', 'PairedPulse', 'InputOutput']
    for p in prots:
        file = 'Y:\eLab\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj+ '\\'+p+'\\CR\\data\\stimlist_CR.csv'
        if os.path.isfile(file):
            stimlist = pd.read_csv(file)
            stimlist = stimlist[stimlist.condition==0]
            stimlist.insert(0,'Prot', p)
            if len(stimlist_hypno) ==0:
                stimlist_hypno  = stimlist
            else:
                stimlist_hypno = pd.concat([stimlist_hypno, stimlist])
    stimlist_hypno = stimlist_hypno.sort_values(by=['date', 'h', 'min', 'StimNum'])
    stimlist_hypno.insert(5,'s', np.random.randint(0, high=59, size=(len(stimlist_hypno),), dtype=int))
    stimlist_hypno = stimlist_hypno[['Prot', 'StimNum','h','s', 'min','ChanP', 'Int_prob', 'date', 'sleep', 'stim_block']] #'s'
    stimlist_hypno = stimlist_hypno.reset_index(drop=True)
    stimlist_hypno.insert(0,'ix', np.arange(len(stimlist_hypno)))
    stimlist_hypno.insert(0,'ix_h', np.arange(len(stimlist_hypno)))
    h0             = stimlist_hypno.h.values[0]
    for day in np.unique(stimlist_hypno.date):
        for h in np.unique(stimlist_hypno.loc[stimlist_hypno.date==day, 'h']):
            m = stimlist_hypno.loc[(stimlist_hypno.h==h)&(stimlist_hypno.date==day), 'min'].values
            s = stimlist_hypno.loc[(stimlist_hypno.h==h)&(stimlist_hypno.date==day), 's'].values
            stimlist_hypno.loc[(stimlist_hypno.h==h)&(stimlist_hypno.date==day), 'ix_h'] = h+m/60+s/3600
    stimlist_hypno.to_csv(file_hypno, index=False,header=True)  #
    #stimlist_hypno.to_csv(file_hypno_all, index=False, header=True)  #

    plt.figure()
    plt.plot(stimlist_hypno.ix, stimlist_hypno.sleep, c=color_elab[0,:], linewidth=2)
    plt.fill_between(stimlist_hypno.ix,stimlist_hypno.sleep,np.zeros((len(stimlist_hypno.ix),))-1, color=color_elab[0,:])
    plt.ylabel('score', fontsize=25)
    plt.yticks([0,1,2,3,4], ['Wake','N1','N2','N3','REM'])
    plt.ylim([-1,5])
    plt.gca().invert_yaxis()
    #plt.xticks(x_tick1,x_ax_h )
    #plt.xticks([])
    plt.tick_params(axis="y", labelsize=18)
    plt.savefig(file_hypno_fig)
    plt.show()
    if update:
        for f in ['InputOutput', 'PairedPulse']:  #
            update_sleep(subj, prot=f, cond_folder='CR')
