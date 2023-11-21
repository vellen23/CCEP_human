import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import h5py
import CCEP_func
import significant_connections as SCF
import freq_funcs as ff
import LL_funcs
import significance_funcs as sig_funcs
import graph_funcs

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

color_sleep = ['#808080', '#919FC7', '#ED936B']
label_sleep = ['Wake', 'NREM', 'REM']

color_sig = ['#808080', '#145da0']
label_sig = ['Non-Sig', 'Sig']

color_elab = np.zeros((3, 3))
color_elab[1, :] = np.array([31, 78, 121]) / 255
color_elab[0, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])

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


### GENERAL functions to analyze characteristics of a CCEP
def plot_SleepState(sc, rc, EEG_resp, con_trial, labels_all, metrics=['LL']):
    num_metrics = len(metrics)
    num_subplots = 3 + num_metrics
    lists = con_trial[(con_trial['Chan'] == rc) & (con_trial['Stim'] == sc)].reset_index(drop=True)
    # Use GridSpec for custom subplot sizes
    # fig = plt.figure(figsize=(6+num_metrics*2.5, 3))
    fig = plt.figure(figsize=(9, 3))
    width_ratios = [2] * 3 + [1] * num_metrics
    gs = gridspec.GridSpec(1, num_subplots, width_ratios=width_ratios)
    axes = [fig.add_subplot(gs[0])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in range(1, 3)]
    axes += [fig.add_subplot(gs[i]) for i in range(3, 4)]
    axes += [fig.add_subplot(gs[i], sharey=axes[3]) for i in range(4, num_subplots)] #
    # axes.append(fig.add_subplot(gs[3]))  # Add the fourth subplot without sharing y-axis

    fig.patch.set_facecolor('xkcd:white')
    sns.set(style='white')
    plt.suptitle(labels_all[sc] + ' -- ' + labels_all[rc], fontsize=10)

    for ax, sleep_lab, sleep_ix in zip(axes, label_sleep, np.arange(len(label_sleep))):
        stimnum = lists.loc[lists.SleepState == sleep_lab, 'Num'].values.astype('int')
        mn = ff.lp_filter(np.mean(EEG_resp[rc, stimnum, :], 0), 30, Fs)
        st = np.std(ff.lp_filter(EEG_resp[rc, stimnum, :], 30, Fs), 0)
        ax.plot(x_ax, mn, color=color_sleep[sleep_ix], linewidth=5, alpha=0.7)
        ax.fill_between(x_ax, mn - st, mn + st, color=color_sleep[sleep_ix], alpha=0.2)
        ax.set_xticks([0, 0.5])
        ax.set_xlabel('time [s]', fontsize = 8)
        if ax is axes[0]:
            ax.set_ylabel('[µV]', fontsize = 8)
            ylim = 50*np.round(np.ceil(np.max(ax.get_ylim()))/50)
            #ax.set_ylim([-ylim, ylim])
            ax.set_yticks([-ylim, 0, ylim])
        else:
            ax.set_yticks([])
        # ax.legend()
        ax.set_title(sleep_lab + ', n: ' + str(len(stimnum)), fontsize = 8)
        ax.axvline(0, color=[0, 0, 0], label='stim')
        ax.set_xlim([-0.25, 0.75])


    # Half-violin, half-strip plot (Raincloud plot)
    for m_ix, metric in enumerate(metrics):
        # Statistics: Wake-NREM, Wake-REM
        _, p_value_NREM, _ = graph_funcs.con_cond_stats(lists.loc[lists.SleepState == 'Wake', metric].values,
                                                        lists.loc[lists.SleepState == 'NREM', metric].values,
                                                        permutation=False, test='MWU')
        _, p_value_REM, _ = graph_funcs.con_cond_stats(lists.loc[lists.SleepState == 'Wake', metric].values,
                                                       lists.loc[lists.SleepState == 'REM', metric].values,
                                                       permutation=False, test='MWU')

        ax = axes[3+m_ix]
        for sleep_state, color in zip(label_sleep, color_sleep):
            #sns.violinplot(x='SleepState', y=metric, data=lists[lists.SleepState == sleep_state], order=label_sleep,
            #                palette=[color], ax=ax, split=True)
            sns.stripplot(x='SleepState', y=metric, data=lists[lists.SleepState == sleep_state], order=label_sleep,
                          palette=[color], linewidth=1, edgecolor='gray', ax=ax, jitter=True, s= 0.5)

            sns.boxplot(x='SleepState', y=metric, data=lists[lists.SleepState == sleep_state], order=label_sleep, palette=[color],
                        fliersize=0, ax=ax)
        # Add significance asterisks
        for ix, p_val in enumerate([p_value_NREM, p_value_REM]):
            if p_val < 0.01:
                ax.text(ix + 1, 0.9*ax.get_ylim()[1], '***', ha='center', va='bottom', fontsize=12, color='black')
            elif p_val < 0.05:
                ax.text(ix + 1, 0.9*ax.get_ylim()[1], '*', ha='center', va='bottom', fontsize=12, color='black')
        ax.set_xticklabels([])
        ax.set_xlabel(metric)
    return axes
