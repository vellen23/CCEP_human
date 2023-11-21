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

####GENERAL functions to analyze characteristics of a CCEP
def plot_CCEP_LL(EEG_resp, con_trial, labels_all, sc_all, rc_all, w_LL = 0.25, Fs = 500, t0 = 1):
    xlim = [-0.3, 0.7]
    ylim_CCEP = [-600, 600]
    ylim_LL = [0, 8]
    fig, axes = plt.subplots(2, len(sc_all), figsize=(len(sc_all)*2.5, 4))  # Create a 1x3 subplot grid
    fig.patch.set_facecolor('xkcd:white')

    for ix, sc, rc in zip(np.arange(len(sc_all)), sc_all, rc_all):
        lists = con_trial[(con_trial['Chan'] == rc) & (con_trial['Stim'] == sc)& (con_trial['Artefact'] <1)].reset_index(drop=True)
        stimnum = lists['Num'].values.astype('int')
        data_CCEP = ff.lp_filter(np.mean(EEG_resp[rc, stimnum, :], 0), 30, Fs)

        # get LL transform
        data_LL = LL_funcs.get_LL_all(np.expand_dims(np.expand_dims(data_CCEP, axis=0), 0), Fs, w_LL)[0,0]

        pk_loc = np.argmax(data_LL[int(t0*Fs):int((t0+0.5)*Fs)])/Fs
        print(pk_loc)
        pk = np.max(data_LL[int(t0*Fs):int((t0+0.5)*Fs)])
        ax = axes[1, ix]
        ax.plot(x_ax+w_LL/2, data_LL, linewidth=2)
        ax.plot(pk_loc+w_LL/2, pk, 'o', color = [1,0,0])
        ax.set_xlabel('time [s]')
        ax.set_xticks([0, 0.5])
        if ix ==0:
            ax.set_ylabel('[uV/ms]')
            ax.set_yticks([0, 3, 6])
        else:
            ax.set_yticks([])
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_LL)
        ax.set_box_aspect(1.5 / 2)
        # plot orignal data with a shadowd window based on peak LL
        ax = axes[0, ix]
        ax.plot(x_ax, data_CCEP, linewidth=2)
        ax.axvspan(pk_loc-w_LL/2, pk_loc+w_LL/2, color = [1,0,0], alpha =0.2)
        ax.set_title(labels_all[sc] + ' - ' + labels_all[rc])
        if ix ==0:
            ax.set_ylabel('[uV]')
            ax.set_yticks([-600,0,600])
        else:
            ax.set_yticks([])
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_CCEP)
        ax.set_xticks([])
        ax.set_box_aspect(1.5 / 2)

    return fig, axes
def plot_CCEP_onset(EEG_resp, con_trial, labels_all, sc_all, rc_all, CC_summ, w_LL = 0.25, Fs = 500, t0 = 1):
    xlim = [-0.25, 0.5]
    ylim_CCEP = [-350, 350]
    ylim_LL = [0, 8]
    fig, axes = plt.subplots(3, len(sc_all), figsize=(len(sc_all)*2.5, 6))  # Create a 1x3 subplot grid
    fig.patch.set_facecolor('xkcd:white')

    for ix, sc, rc in zip(np.arange(len(sc_all)), sc_all, rc_all):
        lists = con_trial[(con_trial['Chan'] == rc) & (con_trial['Stim'] == sc)& (con_trial['Sig'] == 1)& (con_trial['Artefact'] <1)].reset_index(drop=True)
        stimnum = lists['Num'].values.astype('int')
        data_CCEP = ff.lp_filter(np.mean(EEG_resp[rc, stimnum, :], 0), 30, Fs)
        WOI = CC_summ.loc[(CC_summ.Stim == sc) & (CC_summ.Chan == rc), 't_WOI'].values[0]
        delay, data_LL, d1_LL, d2_LL = CCEP_func.cal_delay(data_CCEP, WOI=WOI)
        #data_CCEP = ff.lp_filter(data_CCEP, 30, Fs)
        pk_CCEP_loc = np.argmax(abs(data_CCEP[int(t0 * Fs):int((t0+WOI+0.125) * Fs)]))
        pk_CCEP = data_CCEP[int(t0*Fs+pk_CCEP_loc)]
        # plot second derivative of LL
        ax = axes[2, ix]
        ax.plot(x_ax + w_LL / 2, d2_LL, linewidth=2, alpha=0.5)
        d2_LL[d1_LL < 0] = np.nan  # only increase intresting
        d2_LL[:int((t0 - w_LL / 2) * Fs)] = np.nan  # not before Stim
        d2_LL[int((t0 * Fs) + pk_CCEP_loc):] = np.nan  # not before Stim
        ax.plot(x_ax + w_LL / 2, d2_LL, linewidth=2)
        pk_loc = np.nanargmax(d2_LL[t0 * Fs:]) / Fs
        pk = np.nanmax(d2_LL[t0 * Fs:])
        ax.plot(pk_loc + w_LL / 2, pk, 'o', color=[0, 1, 0])
        ax.set_xlabel('time [s]')
        ax.set_xticks([0, 0.5])
        if ix == 0:
            ax.set_ylabel('[uV/ms]')
            # ax.set_yticks([0, 3, 6])
        else:
            ax.set_yticks([])
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.set_ylim([-0.03, 0.03])
        ax.set_box_aspect(1.5 / 2)
        # Plot LL
        ax = axes[1, ix]
        ax.plot(x_ax+w_LL/2, data_LL, linewidth=2)
        #ax.set_xlabel('time [s]')
        #ax.set_xticks([0, 0.5])
        if ix ==0:
            ax.set_ylabel('[uV/ms]')
            ax.set_yticks([0, 3, 6])
        else:
            ax.set_yticks([])
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_LL)
        ax.axvline(pk_loc + w_LL / 2, color=[0, 1, 0])
        ax.set_box_aspect(1.5 / 2)
        # plot orignal data with a shadowd window based on peak LL
        ax = axes[0, ix]
        ax.plot(x_ax, data_CCEP, linewidth=2)
        ax.axvspan(WOI, WOI+0.25, color = [0,0,0], alpha =0.1)
        ax.set_title(labels_all[sc] + ' - ' + labels_all[rc])
        if ix ==0:
            ax.set_ylabel('[uV]')
            ax.set_yticks([-400,0,400])
        else:
            ax.set_yticks([])
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_CCEP)
        ax.set_xticks([])
        ax.set_box_aspect(1.5 / 2)
        # ax.axvline(pk_loc, color=[0, 1,0])
        ax.axvline(delay, color=[0, 1, 0])
        ax.plot(pk_CCEP_loc/Fs, pk_CCEP, 'o', color=[1, 0, 0])


    return fig, axes

def plot_CCEP_CC(EEG_resp, CC_all, con_trial, labels_all, sc_all, rc_all, w_LL = 0.25, Fs = 500, t0 = 1):
    xlim = [-0.3, 0.7]
    ylim_CCEP = [-600, 600]
    ylim_LL = [0, 12]
    fig, axes = plt.subplots(5, len(sc_all), figsize=(len(sc_all)*2.5, 9))  # Create a 1x3 subplot grid
    fig.patch.set_facecolor('xkcd:white')

    for ix, sc, rc in zip(np.arange(len(sc_all)), sc_all, rc_all):
        lists = con_trial[(con_trial['Chan'] == rc) & (con_trial['Stim'] == sc)& (con_trial['Artefact'] <1)].reset_index(drop=True)
        stimnum = lists['Num'].values.astype('int')
        data_CCEP = ff.lp_filter(np.mean(EEG_resp[rc, stimnum, :], 0), 30, Fs)

        # get LL transform
        data_LL = LL_funcs.get_LL_all(np.expand_dims(np.expand_dims(data_CCEP, axis=0), 0), Fs, w_LL)[0,0]

        pk_loc = np.argmax(data_LL[int(t0*Fs):int((t0+0.3)*Fs)])/Fs
        print(pk_loc)
        pk = np.max(data_LL[int(t0*Fs):int((t0+0.5)*Fs)])
        ax = axes[1, ix]
        ax.plot(x_ax+w_LL/2, data_LL, linewidth=2)
        ax.plot(pk_loc+w_LL/2, pk, 'o', color = [1,0,0])
        if ix ==0:
            ax.set_ylabel('[uV/ms]')
        else:
            ax.set_yticks([])
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_LL)
        ax.set_box_aspect(1.5 / 2)
        # plot orignal data with a shadowd window based on peak LL
        ax = axes[0, ix]
        ax.plot(x_ax, data_CCEP, linewidth=2)
        ax.axvspan(pk_loc-w_LL/2, pk_loc+w_LL/2, color = [0,0,0], alpha =0.2)
        ax.text(pk_loc - w_LL / 2, -500, 'WOI')
        ax.set_title(labels_all[sc] + ' - ' + labels_all[rc])
        if ix ==0:
            ax.set_ylabel('[uV]')
            ax.set_yticks([-600,0,600])
        else:
            ax.set_yticks([])
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_CCEP)
        ax.set_xticks([])
        ax.set_box_aspect(1.5 / 2)
        #### 3. plot two cluster centers
        ax = axes[2, ix]
        for ix_CC in [1,2]:
            ax.plot(x_ax, CC_all[sc, rc, ix_CC], linewidth=2, label = 'CC'+str(ix_CC), color = color_elab[ix_CC])
        ax.axvspan(pk_loc - w_LL / 2, pk_loc + w_LL / 2, color=[0, 0, 0], alpha=0.2)
        ax.set_xticks([])
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.legend()
        ax.set_ylim(ylim_CCEP)
        ax.set_xticks([])
        ax.set_box_aspect(1.5 / 2)
        if ix ==0:
            ax.set_ylabel('[uV]')
            ax.set_yticks([-600,0,600])
        else:
            ax.set_yticks([])

        #### 4. LL transform of CC
        ax = axes[3, ix]
        ax_hist = axes[4, ix]
        ylim = 3
        for ix_CC in [1, 2]:
            data_LL = LL_funcs.get_LL_all(np.expand_dims(np.expand_dims(CC_all[sc, rc, ix_CC], axis=0), 0), Fs, w_LL)[0, 0]
            ax.plot(x_ax + w_LL / 2, data_LL, linewidth=2, label = 'CC'+str(ix_CC), color = color_elab[ix_CC])
            pk = data_LL[int((t0+pk_loc)*Fs)]
            ylim = np.max([ylim, np.ceil(pk)])
            ax_hist.axvline(pk, label = 'CC'+str(ix_CC), color = color_elab[ix_CC])
            ax.plot(pk_loc + w_LL / 2, pk, 'o', color=[1, 0, 0])
        ax.legend()
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_LL)
        ax.set_xticks([])
        ax.set_box_aspect(1.5 / 2)
        if ix ==0:
            ax.set_ylabel('[uV/ms]')
        else:
            ax.set_yticks([])
        ax.set_xlabel('time [s]')
        ax.set_xticks([0, 0.5])

        # 5. surr distribution
        LL_surr, _, _, _ = SCF.get_CC_surr(rc, con_trial, EEG_resp, 100)  # return LL_surr[1:, :], LL_surr_data[1:, :, :]
        ax_hist.hist(LL_surr.flatten(), color = [0,0,0], alpha = 0.5)
        ax_hist.set_xlim([0,ylim])
        ax_hist.set_xlabel('[uV/ms]')
        ax_hist.set_ylabel('Surrogate')
        ax_hist.axvline(np.percentile(LL_surr, 95), color = [1,0,0], label='CC' + str(ix_CC))
        ax_hist.set_box_aspect(1.5 / 2)
    return fig, axes

def plot_trial_test(EEG_resp, CC_all, surr_data, trials, sc, rc,t_WOI, w_LL = 0.25, Fs = 500, t0 = 1):
    xlim = [-0.3, 0.7]
    ylim_CCEP = [-600, 600]
    ylim_LL = [0, 12]
    fig, axes = plt.subplots(5, 4, figsize=(10, 9))  # Create a 1x3 subplot grid
    fig.patch.set_facecolor('xkcd:white')
    # 1. plot both Cluster Centers
    for ix_CC in [1, 2]:
        ax = axes[0, ix_CC-1]
        ax.plot(x_ax, CC_all[sc, rc, ix_CC], linewidth=2, color=color_elab[ix_CC])
        ax.set_xticks([])
        ax.axvline(0, color=[0, 0, 0])
        ax.set_xlim(xlim)
        ax.set_title('CC' + str(ix_CC))
        ax.set_ylim(ylim_CCEP)
        ax.set_xticks([])
        ax.set_box_aspect(1.5 / 2)
        ax.axvspan(t_WOI, t_WOI + w_LL, color=[0, 0, 0], alpha=0.1)

    # 2. plot surrogate distribution
    # Merging the last two subplots in the first row
    gs = axes[0, 2].get_gridspec()
    # remove the underlying axes
    for ax in axes[0, 2:]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, 2:])
    axbig.hist(surr_data)  # Assuming surr_data is in the correct format
    axbig.axvline(np.percentile(surr_data, 90), color=[1,0,0])
    axbig.set_title('Surrogate Distribution')

    # 3. for each trial plot 1) the signal 2) the LL transform 3) the pearson corr to both CC and 4) the correcponding p^2 *LL for both CC
    for ix_trial, trial in enumerate(trials):
        # get trial specifc data
        data_CCEP = ff.lp_filter(EEG_resp[rc, trial, :], 45, Fs)
        ax = axes[1, ix_trial]
        ax.plot(x_ax, data_CCEP)
        ax.set_box_aspect(1.5 / 2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_CCEP)
        ax.axvline(0, color=[0, 0, 0])
        ax.axvspan(t_WOI, t_WOI+w_LL, color=[0, 0, 0], alpha = 0.1)
        # get LL transform
        data_LL = LL_funcs.get_LL_all(np.expand_dims(np.expand_dims(data_CCEP, axis=0), 0), Fs, w_LL)[0, 0]
        ax = axes[2, ix_trial]
        ax.set_box_aspect(1.5 / 2)
        ax.set_xlim(xlim)
        ax.plot(x_ax+w_LL/2, data_LL)
        ax.set_ylim(ylim_LL)
        ax.axvline(0, color=[0, 0, 0]) #t_WOI+0.25
        ax.axvline(t_WOI+w_LL, color=[0, 0, 0])
        # correlation to both CC (in specific WOI)
        ax_corr = axes[3, ix_trial]
        ax_comb = axes[4, ix_trial]
        ax_corr.set_box_aspect(1.5 / 2)
        ax_corr.set_ylim([-1.1, 1.1])
        ax_comb.set_box_aspect(1.5 / 2)
        ax_comb.set_xlim(xlim)
        ax_corr.set_xlim(xlim)
        ax_comb.axvline(0, color=[0, 0, 0])
        ax_comb.set_ylim(ylim_LL)
        ax_corr.axvline(0, color=[0, 0, 0])
        ax_corr.axvline(t_WOI + w_LL, color=[0, 0, 0])
        ax_comb.axvline(t_WOI + w_LL, color=[0, 0, 0])
        ax_comb.axhline(np.percentile(surr_data, 90), color=[1,0,0])
        for ix_CC in [1, 2]:
            corr = sig_funcs.get_pearson2mean_windowed(CC_all[sc, rc,ix_CC], np.expand_dims(data_CCEP,0), t0+t_WOI, 0.25, 500 )[0]
            ax_corr.plot(x_ax +w_LL/2, corr, color=color_elab[ix_CC])
            ax_comb.plot(x_ax +w_LL/2, np.sign(corr) *corr**2*data_LL, color=color_elab[ix_CC])
    return fig, axes


def plot_SigCon_examples(sc_all, rc_all, con_trial, EEG_resp, labels_all, thesis=1):
    if thesis:
        fig, axes = plt.subplots(1, len(sc_all), figsize=(8, 2))  # Adjust the figure size
        ylim_ccep = [-600, 300]
    else:
        fig, axes = plt.subplots(1, len(sc_all), figsize=(15, 5))
        ylim_ccep = [-600, 300]
    for i, (sc, rc) in enumerate(zip(sc_all, rc_all)):
        num_all = np.unique(
            con_trial.loc[(con_trial.Chan == rc) & (con_trial.Stim == sc) & (con_trial.Artefact < 1), 'Num'].values)
        mean = ff.lp_filter(np.mean(EEG_resp[rc, num_all], 0), 30, Fs)
        resp_zscore_mean = CCEP_func.zscore_CCEP(mean, t_0=1, w0=0.5, Fs=500)

        ax1 = axes[i]
        ax1.plot(x_ax, mean, label='Signal 1', color='k')
        ax1.set_xlabel('Time [s]')
        if i == 0:
            ax1.set_ylabel('Mean CCEP [uV]', color='k')

        else:
            ax1.set_ylabel('', color='k')

        ax1.set_yticks([-250, 0, 250])
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.set_ylim(ylim_ccep)

        ax1.set_xlim([-0.5, 1.5])  # Set x-axis limits

        # Create the second y-axis (right)
        ax2 = ax1.twinx()
        ax2.plot(x_ax, resp_zscore_mean, label='Mean z-score', color=color_elab[0])

        ax2.tick_params(axis='y', labelcolor=color_elab[0])
        ax2.set_ylim([-15, 35])

        if i == len(sc_all) - 1:
            ax2.set_ylabel('Z-score of Mean', color=color_elab[0])

        else:
            ax2.set_ylabel('', color='k')
        ax2.set_yticks([-10, 0, 10])
        ax2.hlines(y=-6, xmin=0.05, xmax=0.5, linewidth=2, color='r', alpha=0.2)
        ax2.hlines(y=6, xmin=0.05, xmax=0.5, linewidth=2, color='r', alpha=0.2)

        # Add a title to each subplot
        ax1.set_title(labels_all[sc] + ' -- ' + labels_all[rc])

        # Add vertical line at x=0
        ax1.axvline(0, color=[1, 0, 0])

    return fig, axes


