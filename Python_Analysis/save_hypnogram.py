import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
root = Tk()
root.withdraw()

subj            = "EL012"
#subjs = ['EL003', 'EL004', 'EL005', 'EL010', 'EL011']
subjs = ["EL017"]
for subj in subjs:
    cwd             = os.getcwd()
    print(subj)
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
