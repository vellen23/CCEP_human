import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import h5py
import freq_funcs as ff

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]

# General Info based on how data was epoched
Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))

color_sleep = ['#808080', '#145da0', '#ff1919']
label_sleep = ['Wake', 'NREM', 'REM']

color_sig = ['#808080', '#145da0']
label_sig = ['Non-Sig', 'Sig']

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

####GENERAL functions to analyze characteristics of a CCEP
def plot_SleepState(sc, rc, EEG_resp, con_trial, labels_all):
    lists = con_trial[(con_trial['Chan'] == rc) & (con_trial['Stim'] == sc)].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey= True)  # Create a 1x3 subplot grid

    fig.patch.set_facecolor('xkcd:white')
    sns.set(style='white')
    plt.suptitle(labels_all[sc] + ' -- ' + labels_all[rc], fontsize=16)

    for ax, sleep_lab, sleep_ix in zip(axes, label_sleep, np.arange(len(label_sleep))):
        stimnum = lists.loc[lists.SleepState == sleep_lab, 'Num'].values.astype('int')
        mn = ff.lp_filter(np.mean(EEG_resp[rc, stimnum, :], 0), 30, Fs)
        st = np.std(ff.lp_filter(EEG_resp[rc, stimnum, :], 30, Fs), 0)
        ax.plot(x_ax, mn, color=color_sleep[sleep_ix], linewidth=5, alpha=0.7, label=sleep_lab + ', n: ' + str(len(stimnum)))
        ax.fill_between(x_ax, mn - st, mn + st, color=color_sleep[sleep_ix], alpha=0.2)
        ax.set_xticks([-0.5, 0, 0., 1])
        ax.set_xlabel('time [s]')
        ax.set_ylabel('[uV]')
        ax.legend()
        ax.axvline(0, color=[0, 0, 0], label='stim')
        ax.set_xlim([-0.5, 1])

    return axes

def plot_Sig(sc, rc, EEG_resp, con_trial, labels_all):
    lists = con_trial[(con_trial['Chan'] == rc) & (con_trial['Stim'] == sc)].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey= True)  # Create a 1x3 subplot grid

    fig.patch.set_facecolor('xkcd:white')
    sns.set(style='white')
    plt.suptitle(labels_all[sc] + ' -- ' + labels_all[rc], fontsize=16)

    for ax, sig_lab, sig_ix in zip(axes, label_sig, np.arange(len(label_sig))):
        stimnum = lists.loc[lists.Sig == sig_ix, 'Num'].values.astype('int')
        mn = ff.lp_filter(np.mean(EEG_resp[rc, stimnum, :], 0), 30, Fs)
        st = np.std(ff.lp_filter(EEG_resp[rc, stimnum, :], 30, Fs), 0)
        ax.plot(x_ax, mn, color=color_sig[sig_ix], linewidth=5, alpha=0.7, label=sig_lab + ', n: ' + str(len(stimnum)))
        ax.fill_between(x_ax, mn - st, mn + st, color=color_sig[sig_ix], alpha=0.2)
        ax.set_xticks([-0.5, 0, 0., 1])
        ax.set_xlabel('time [s]')
        ax.set_ylabel('[uV]')
        ax.legend()
        ax.axvline(0, color=[0, 0, 0], label='stim')
        ax.set_xlim([-0.5, 1])

    return axes


