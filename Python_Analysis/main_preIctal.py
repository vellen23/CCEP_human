import os
import sys
from datetime import datetime, timedelta
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append('T:\EL_experiment\Codes\CCEP_human\Python_Analysis\py_functions')
import basic_func as bf

# Configuration and global variables
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab
chan_labels = ['SOZ', 'Propagation', 'uninvolved']
chan_labels = ['ictogenic tissue', 'non-ictogenic tissue']
plt.rcParams.update({
    'font.family': 'arial',
    'font.size': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'svg.fonttype': 'none',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 10
})

def run_subj(subj,folder = 'BrainMapping'):
    path_patient_analysis = sub_path + '\EvM\Projects\EL_experiment\Analysis\Patients\\' + subj
    path_gen = os.path.join(sub_path + '\Patients\\' + subj)
    if not os.path.exists(path_gen):
        path_gen = 'T:\\EL_experiment\\Patients\\' + subj
    path_infos = os.path.join(path_gen, 'Electrodes')
    # labels
    files_list = glob(path_patient_analysis + '\\' + folder + '/data/Stim_list_*')
    i = 0
    stimlist = pd.read_csv(files_list[i])
    lbls = pd.read_excel(os.path.join(path_infos, subj + "_labels.xlsx"), header=0, sheet_name='BP')
    if 'type' in lbls:
        lbls = lbls[lbls.type == 'SEEG']
        lbls = lbls.reset_index(drop=True)
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist,
        lbls)
    con_trial, sz_log = get_data(subj, path_patient_analysis)
    con_trial, con_sig = clean_table(con_trial)
    # Define the directory path
    directory_path = os.path.join(path_patient_analysis, 'preIctal', 'figures')

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    sz_table = get_con_sz(subj, con_trial, sz_log, lbls, con_sig, path_patient_analysis)
    sz_table.insert(0,'Subj', subj)
    sz_table.to_csv(os.path.join(path_patient_analysis, 'preIctal', 'table_preIctal.csv'), header=True, index =False)
    print(subj+' -- DONE')

def get_data(subj,path_patient_analysis, cond_folders = ['BrainMapping', 'InputOutput', 'Pairedpulse']):
    # get conenction trials
    con_trial_list = []  # Create a list to store DataFrames
    for cf in cond_folders:
        file_con = os.path.join(path_patient_analysis, cf, 'CR', 'data', 'con_trial_all.csv')
        if os.path.isfile(file_con):
            print(cf)
            con_trial_list.append(pd.read_csv(file_con))

    # Concatenate all DataFrames in the list into one big DataFrame
    con_trial = pd.concat(con_trial_list, ignore_index=True)
    con_trial.loc[np.isnan(con_trial.Int), 'Int'] = 3 # Brain Map does not have Int value yet.
    if "Intp" in con_trial:
        con_trial.loc[(con_trial.IPI >= 0), 'Int'] = con_trial.loc[(con_trial.IPI >= 0), 'Intp']

    for new_col in ['Intc', 'IPI']:
        if new_col in con_trial:
            con_trial.loc[np.isnan(con_trial[new_col]), new_col] = 0
        else:
            con_trial[new_col] = 0
    # remove artefact
    con_trial = con_trial[(con_trial.Artefact < 1)].reset_index(drop=True)

    # if 'IPI' in con_trial.columns:  # set probing Pulse as Int bvalue for PP protocol (only for fake SP stimulations where IPI >500ms)
    #     con_trial.loc[(con_trial.IPI == 0), 'Int'] = con_trial.loc[(con_trial.IPI == 0), 'Intp']
    #     con_trial.loc[np.isnan(con_trial.IPI), 'IPI'] = 0
    #     con_trial = con_trial[(con_trial.IPI == 0) & (con_trial.Artefact <2 )].reset_index(drop=True)
    # else:
    #     con_trial = con_trial[(con_trial.Artefact <2)].reset_index(drop=True)
    # get seizure data
    sz_log = pd.read_excel(os.path.join(sub_path, 'Patients', subj, 'Data', 'EL_experiment', 'SZ_log.xlsx'))
    sz_log = sz_log[~np.isnan(sz_log.SZ)].reset_index(drop=True)
    if "sel" in sz_log:
        sz_log = sz_log[sz_log.sel==1].reset_index(drop=True)

    return con_trial, sz_log

def clean_table(con_trial):
    con_trial['Con_ID'] = con_trial.groupby(['Stim', 'Chan']).ngroup()
    # find significant connections
    con_summary = con_trial.groupby(['Con_ID', 'Stim', 'Chan'], as_index=False)['Sig'].mean()
    con_summary = con_summary[con_summary.Sig > 0.8].reset_index(drop=True)
    con_sig = con_summary.Con_ID.values
    return con_trial, con_sig

def get_con_preictal(con_trial, time_sz,lbls, SOZ_label, con_sig, precital_h = 4, postictal_h = 0.05):
    # select only trials that are 1h before seizure time
    trials_preictal = con_trial[
        ((pd.to_datetime(con_trial['Time']) - time_sz) > timedelta(seconds=-precital_h * 3600)) & (
                (pd.to_datetime(con_trial['Time']) - time_sz) < timedelta(
            seconds=postictal_h * 3600))].reset_index(drop=True)

    # normalize LL specific to intensity and conditioning based 4h before SZ
    trials_preictal['LL_z'] = trials_preictal.groupby(['Stim', 'Chan', 'Int', 'Intc', 'IPI'], group_keys=False)['LL'].apply(
        lambda x: (x - x.mean()) / x.std())

    # Create a new column 'TimeDiffSeconds' in the DataFrame
    trials_preictal['TtSZ'] = np.floor(
        ((pd.to_datetime(trials_preictal['Time']) - time_sz).dt.total_seconds()) / 60)
    # Create a new column 'TimeDiffSeconds' in the DataFrame
    trials_preictal['TtSZ_s'] = np.round(
        ((pd.to_datetime(trials_preictal['Time']) - time_sz).dt.total_seconds()))
    # only 30min pre seizure
    trials_preictal = trials_preictal[(trials_preictal.TtSZ_s>-1201)&(trials_preictal.TtSZ_s<0)].reset_index(drop=True)

    # normalize by time (channel specific)
    trials_preictal['LL_norm'] = trials_preictal.groupby(['Chan'], group_keys=False)['LL_z'].apply(
        lambda x: (x - x.mean()) / x.std())

    # timing
    trials_preictal.insert(5, 'PreIctal', '>7min')
    trials_preictal.loc[(trials_preictal.TtSZ < -2) & (trials_preictal.TtSZ > -8), 'PreIctal'] = '>2min'
    trials_preictal.loc[(trials_preictal.TtSZ < 0) & (trials_preictal.TtSZ > -3), 'PreIctal'] = '>0min'
    trials_preictal.loc[(trials_preictal.TtSZ > -1), 'PreIctal'] = 'postIctal'
    # connections
    trials_preictal.insert(0, 'SOZ', 'non-sig')
    trials_preictal = trials_preictal[np.isin(trials_preictal.Con_ID, con_sig)].reset_index(drop=True)
    chan_ictal = np.where(lbls[SOZ_label] >0)[0]
    chan_nonictal = np.where(lbls[SOZ_label] == 0)[0]
    trials_preictal.loc[(np.isin(trials_preictal.Chan, chan_ictal)), 'SOZ'] = chan_labels[0]
    trials_preictal.loc[(np.isin(trials_preictal.Chan, chan_nonictal)), 'SOZ'] = chan_labels[1]
    trials_preictal = trials_preictal[trials_preictal.SOZ != 'non-sig'].reset_index(drop=True)
    # for chan_label_ix in range(len(chan_labels)):
    #     chan_sel = np.where(lbls[SOZ_label] == chan_label_ix)[0]
    #     trials_preictal.loc[np.isin(trials_preictal.Con_ID, con_sig) & np.isin(trials_preictal.Chan, chan_sel), 'SOZ'] = \
    #     chan_labels[chan_label_ix]
    keep_col = ['Con_ID', 'Stim', 'Chan', 'Int', 'TtSZ', 'TtSZ_s', 'LL_norm', 'LL_z', 'SOZ', 'PreIctal', 'IPI']
    trials_preictal = trials_preictal[keep_col].reset_index(drop=True)
    return trials_preictal
    #
def plot_figures(trials_preictal, path, title, filename, filename_dynamic):
    # Plot
    sns.catplot(x='SOZ', y='LL_norm', hue='PreIctal', order=chan_labels,
                data=trials_preictal[(trials_preictal.PreIctal != 'postIctal') & (trials_preictal.SOZ != 'non-sig')],
                kind='box', height=3, aspect=1.5,
                palette='Blues')
    plt.title(title)
    plt.ylabel('Normalized LL')
    ## X:\4 e-Lab\EvM\Projects\EL_experiment\Analysis\Patients\EL012\preIctal\figures
    plt.savefig(os.path.join(path, filename))
    data_sel = trials_preictal[(trials_preictal.SOZ != 'non-sig')]
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 1, figsize=(5, 3))
    plt.suptitle(title)
    sns.lineplot(x='TtSZ', y='LL_norm', hue='SOZ', data=data_sel, ax=axes, palette='Reds', hue_order=chan_labels)
    axes.set_xlabel('')
    axes.set_ylabel('LL z-scored')
    axes.set_title('Mean int-normalized response')
    axes.axvline(-7, color=[0, 0, 0])
    axes.axvline(-2, color=[0, 0, 0])
    axes.axvline(0, color=[1, 0, 0])
    axes.axhline(0, color=[0, 0, 0])
    axes.set_xlim([-20, 4])
    axes.set_ylim([-3, 5])
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename_dynamic))

def plot_figures_zoom(trials_preictal, path, title, filename, filename_dynamic):
    # Plot
    sns.catplot(x='SOZ', y='LL_norm', hue='PreIctal', order=chan_labels,
                data=trials_preictal[(trials_preictal.PreIctal != 'postIctal') & (trials_preictal.SOZ != 'non-sig')],
                kind='box', height=3, aspect=1.5,
                palette='Blues')
    plt.title(title)
    plt.ylabel('Normalized LL')
    ## X:\4 e-Lab\EvM\Projects\EL_experiment\Analysis\Patients\EL012\preIctal\figures
    plt.savefig(os.path.join(path, filename))

    data_sel = trials_preictal[(trials_preictal.SOZ != 'non-sig')]
    win_zoom = [-20, 3]
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(2, 1, figsize=(5, 6))
    plt.suptitle(title)
    # Plot the first figure on the left subplot
    axes[0].axvspan(win_zoom[0], win_zoom[1], color=[0, 0, 0], alpha=0.05)
    sns.lineplot(x='TtSZ', y='LL_norm', hue='SOZ', data=data_sel, ax=axes[0], palette='Reds', hue_order=chan_labels)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('LL z-scored')
    axes[0].set_title('Mean int-normalized response')
    axes[0].axvline(-10, color=[0, 0, 0])
    axes[0].axvline(-5, color=[0, 0, 0])
    axes[0].axvline(0, color=[1, 0, 0])
    axes[0].axhline(0, color=[0, 0, 0])
    axes[0].set_xlim([-60, 4])
    axes[0].set_ylim([-3, 5])

    # Plot the second figure on the right subplot
    sns.lineplot(x='TtSZ', y='LL_norm', hue='SOZ', data=data_sel, ax=axes[1], palette='Reds', hue_order=chan_labels)
    axes[1].set_xlabel('Time to Seizure [min]')
    axes[1].set_ylabel('LL z-scored')
    axes[1].set_title('Zoom')
    axes[1].axvline(0, color=[1, 0, 0])
    axes[1].set_xlim(win_zoom)
    axes[1].set_ylim([-3, 5])
    axes[1].axhline(0, color=[0, 0, 0])
    axes[1].axvline(-5, color=[0, 0, 0])

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename_dynamic))

def get_con_sz(subj, con_trial, sz_log, lbls, con_sig, path_patient_analysis, plot = 1):
    # repeat for each seizure
    path = os.path.join(path_patient_analysis, 'preIctal', 'figures')
    for ix_sz in range(len(sz_log)):
        time_sz = datetime.combine(sz_log.Date[ix_sz].to_pydatetime().date(), sz_log.Time[ix_sz])
        SOZ_label = sz_log.SOZ_label[ix_sz]

        trials_preictal = get_con_preictal(con_trial, time_sz,lbls, SOZ_label, con_sig, precital_h = 1, postictal_h = 0.05)
        trials_preictal['SZ'] = ix_sz+1
        if plot:
            filename = 'sz_boxplot_' + str(ix_sz + 1) + '.svg'
            filename2 = 'sz_dynamics_' + str(ix_sz + 1) + '.svg'
            title = subj + ' -- SZ#' + str(ix_sz + 1)
            plot_figures(trials_preictal, path, title, filename, filename2)
        if ix_sz == 0 :
            sz_table = trials_preictal
        else:
            sz_table = pd.concat(([sz_table,trials_preictal ]))
    if ix_sz>0:
        if plot:
            filename = 'sz_boxplot_all.svg'
            filename2 = 'sz_dynamics_all.svg'
            title = subj + ' -- all SZ'
            plot_figures(sz_table, path, title, filename, filename2)

    return sz_table

subjs = ['EL012', 'EL024', 'EL027','EL015', 'EL019', 'EL013', 'EL014']
for subj in subjs:
    run_subj(subj)