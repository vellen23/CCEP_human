import os
import numpy as np
import sklearn
import freq_funcs as ff
import LL_funcs as LLf
import scipy
from sklearn.metrics import auc
import pandas as pd
import CCEP_func

# regions         = pd.read_excel("T:\EL_experiment\Patients\\" +'all'+"\elab_labels.xlsx", sheet_name='regions', header=0)
# color_regions   = regions.color.values
# regions         = regions.label.values
cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]
SleepStates_val = ['Wake', 'NREM', 'REM']
Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3

# dur[0,:]       = np.int32(np.sum(abs(dur)))
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])


def get_AUC_MAX_Pearson(Int_values, LL_values):
    AUC = auc(Int_values, LL_values)
    MAX = np.mean(
        np.sort(LL_values)[-3:])
    rho = scipy.stats.pearsonr(Int_values, LL_values)[0]
    return AUC, MAX, rho


def get_AUC_surr(rc, con_trial, EEG_resp, mx_true, Int_selc, n_trial=100, n=10, w=0.25):
    AUC_surr = np.zeros((n * 3,))
    max_surr = np.zeros((n * 3,))
    p_surr = np.zeros((n * 3,))
    stim_trials = np.unique(
        con_trial.loc[(con_trial.Stim == rc) | (con_trial.Stim == rc - 1) | (con_trial.Stim == rc + 1), 'Num'])
    trials_all = np.unique(con_trial.loc[(con_trial.Chan == rc) & (con_trial.Artefact < 1), 'Num'])
    trials_all = np.array([i for i in trials_all if i not in stim_trials])
    trials_all = trials_all.astype('int')
    Int_norm = (Int_selc - np.min(Int_selc)) / (np.max(Int_selc) - np.min(Int_selc))
    for rep in range(n):
        mx_all = np.zeros((len(Int_selc), 3))
        for i, intensity in enumerate(Int_selc):
            num_sel = np.unique(np.random.choice(trials_all, n_trial, replace=False))
            resp = ff.lp_filter(np.nanmean(EEG_resp[rc, num_sel, :], 0), 45, Fs)
            LL_resp = LLf.get_LL_all(np.expand_dims(resp, [0, 1]), Fs, w)[0][0]
            mx_all[i, 0] = np.max(LL_resp[int((t0 - 0.5) * Fs):int((t0 - w / 2) * Fs)])
            mx_all[i, 1] = np.max(LL_resp[int((w / 2) * Fs):int((t0 - 0.5) * Fs)])
            mx_all[i, 2] = np.max(LL_resp[int((t0 + 2) * Fs):int((t0 + 25) * Fs)])
        for i in range(3):
            normalized_mx_BL = (mx_all[:, i] - np.min(mx_all[:, i])) / (np.max(mx_true) - np.min(mx_all[:, i]))

            AUC_surr[int((i * n) + rep)], max_surr[int((i * n) + rep)], p_surr[
                int((i * n) + rep)] = get_AUC_MAX_Pearson(Int_norm, normalized_mx_BL)

    return AUC_surr, max_surr, p_surr


def get_pvalue(real_array, surr_array):
    count = np.sum(surr_array <= real_array[:, np.newaxis], axis=1)

    # Calculate the p-values
    p_values = np.array(count) / len(surr_array)

    return p_values


def get_AUC_real(sc, rc, con_trial, EEG_resp, n=10, w=0.25):
    dat = con_trial[(con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (con_trial['Artefact'] < 1)].reset_index(
        drop=True)
    Int_selc = np.unique(dat['Int'])
    n_trials = 200
    mx_all = np.zeros((len(Int_selc), 2))
    for i, intensity in enumerate(Int_selc):
        dati = dat[(dat['Int'] == intensity) & (dat['Artefact'] < 1)].reset_index(drop=True)
        resp = ff.lp_filter(np.nanmean(EEG_resp[rc, dati.Num.values.astype('int'), :], 0), 45, Fs)
        LL_resp = LLf.get_LL_all(np.expand_dims(resp, [0, 1]), Fs, w)[0][0]
        mx = np.max(LL_resp[int((t0 + w / 2) * Fs):int((t0 + 0.5 + w / 2) * Fs)])
        mx_all[i, 0] = mx
        n_trials = np.min([n_trials, len(dati.Num.values.astype('int'))])
    mx_all = np.array(mx_all[:, 0])
    mx_norm = (mx_all - np.min(mx_all)) / (np.max(mx_all) - np.min(mx_all))
    Int_norm = (Int_selc - np.min(Int_selc)) / (np.max(Int_selc) - np.min(Int_selc))
    AUC, MAX, rho = get_AUC_MAX_Pearson(Int_norm, mx_norm)
    # get surrogate data
    AUC_surr, max_surr, rho_surr = get_AUC_surr(rc, con_trial, EEG_resp, mx_all, Int_selc, n_trial=n_trials, n=n,
                                                w=0.25)
    AUC_p = get_pvalue(np.array([AUC]), AUC_surr)[0]
    MAX_p = get_pvalue(np.array([MAX]), max_surr)[0]
    rho_p = get_pvalue(np.array([rho]), rho_surr)[0]

    return AUC, MAX, rho, AUC_p, MAX_p, rho_p


def save_AUC_connection(con_trial, EEG_resp):
    import matplotlib.pyplot as plt
    stim_all = np.unique(con_trial['Stim'])
    chan_all = np.unique(con_trial['Chan'])
    data_rows = []  # Use for collecting rows of data
    for sc in stim_all.astype('int'):
        for rc in chan_all.astype('int'):
            dat = con_trial[
                (con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (con_trial['Artefact'] < 1)].reset_index(
                drop=True)
            if len(dat) > 0:
                AUC, MAX, rho, AUC_p, MAX_p, rho_p = get_AUC_real(sc, rc, con_trial, EEG_resp, n=40, w=0.25)
                # get delay
                num = np.unique(
                    con_trial.loc[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1), 'Num'])
                trials = EEG_resp[rc, num, :]
                WOI = 0.1
                peak_lat, _, _ = CCEP_func.peak_latency(trials, WOI, t0=1, Fs=500, w_LL=0.25)
                # Append each set of results as a new row in the data_rows list
                data_rows.append([sc, rc, AUC, MAX, rho, 1 - AUC_p, 1 - MAX_p, 1 - rho_p])

    # Create the DataFrame after the loop, using the collected rows
    df = pd.DataFrame(data_rows, columns=["Stim", "Chan", "AUC", "MAX", "rho", "AUC_p", "MAX_p", "rho_p"])
    return df


def get_delay(con_trial, EEG_resp, auc_summary, plot=0):
    import matplotlib.pyplot as plt
    Fs = 500
    dur = np.zeros((1, 2), dtype=np.int32)
    t0 = 1
    dur[0, 0] = -t0
    dur[0, 1] = 3

    auc_summary['peak_latency'] = np.nan
    auc_summary['onset'] = np.nan
    df_auc_sig = auc_summary.loc[
        (auc_summary.AUC_p < 0.01) & (auc_summary.MAX_p < 0.01) & (auc_summary.rho_p < 0.01)].reset_index(drop=True)
    stim_all = np.unique(df_auc_sig['Stim'])

    for sc in stim_all.astype('int'):
        chan_all = np.unique(df_auc_sig.loc[(df_auc_sig.Stim == sc), 'Chan'])
        for rc in chan_all.astype('int'):
            dat = con_trial[
                (con_trial['Int'] > 2) & (con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (
                            con_trial['Artefact'] < 1)].reset_index(
                drop=True)
            if (len(dat) > 0):
                num = np.unique(dat['Num'])
                trials = EEG_resp[rc, num, :]
                WOI = 0.1
                t_onset, peak_lat, polarity, peak_detected = CCEP_func.CCEP_onset(trials, WOI=WOI, t0=1, Fs=500,
                                                                                  w_LL=0.25, plot=False,
                                                                                  skip_nonpeak=True)
                # peak_lat, _, _ = CCEP_func.peak_latency(trials, WOI, t0=1, Fs=500, w_LL=0.25)
                # Append each set of results as a new row in the data_rows list
                if peak_detected:
                    auc_summary.loc[
                        (auc_summary['Stim'] == sc) & (auc_summary['Chan'] == rc), 'peak_latency'] = peak_lat
                    auc_summary.loc[(auc_summary['Stim'] == sc) & (auc_summary['Chan'] == rc), 'onset'] = t_onset
                if plot:
                    plt.plot(x_ax, np.mean(trials, 0))
                    plt.xlim([-0.5, 1])
                    plt.axvline(0, color='k')
                    plt.axvline(peak_lat, color='r')
                    plt.show()
    return auc_summary


def save_AUC_connection_SS(con_trial, EEG_resp):
    stim_all = np.unique(con_trial['Stim'])
    chan_all = np.unique(con_trial['Chan'])
    data_rows = []  # Use for collecting rows of data
    sleep_states = ['Wake', 'NREM', 'REM']
    for sc in stim_all.astype('int'):
        for rc in chan_all.astype('int'):
            for ss in sleep_states:
                dat = con_trial[
                    (con_trial['SleepState'] == ss) & (con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (
                            con_trial['Artefact'] < 1)].reset_index(
                    drop=True)
                if len(dat) > 0:
                    AUC, MAX, rho, AUC_p, MAX_p, rho_p = get_AUC_real(sc, rc, con_trial, EEG_resp, n=40, w=0.25)
                    # Append each set of results as a new row in the data_rows list
                    data_rows.append([sc, rc, AUC, MAX, rho, 1 - AUC_p, 1 - MAX_p, 1 - rho_p])

    # Create the DataFrame after the loop, using the collected rows
    df = pd.DataFrame(data_rows, columns=["Stim", "Chan", "AUC", "MAX", "rho", "AUC_p", "MAX_p", "rho_p"])
    return df


def get_AUC_trials(sc, rc, con_trial, EEG_resp, cond_val, n_trial=3, n_shuffle=10, w=0.25):
    data_pd = []
    Int_all = np.unique(con_trial.Int)
    for j in range(len(cond_val)):
        for i in range(len(Int_all)):
            count_run = 0
            Int = Int_all[i]
            dat = con_trial[
                (con_trial.Artefact < 1) & (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Int == Int) & (
                        con_trial.SleepState == cond_val[j])]
            stimNum = dat.Num.values.astype('int')
            n_epoch = np.floor(len(stimNum) / n_trial).astype('int')
            for n in range(n_shuffle):
                if n > 0:  # shuffle
                    np.random.shuffle(stimNum)
                for ne in range(n_epoch):
                    num_sel = np.sort(stimNum[int(ne * n_trial):int((ne + 1) * n_trial)])
                    # n_trial_spec = np.min([n_trial, len(stimNum)])
                    # num_sel = np.unique(np.random.choice(stimNum, n_trial_spec, replace = False))
                    mn = ff.lp_filter(np.mean(EEG_resp[rc, num_sel, int(t0 * Fs):int((t0 + 0.5) * Fs)], 0), 45, Fs)
                    LL_mn = LLf.get_LL_all(np.expand_dims(mn, [0, 1]), Fs, w)[0][0]
                    # plt.scatter(Int, np.max(LL_mn), color = color[j])
                    data_pd.append([Int, count_run, SleepStates_val[j], np.max(LL_mn)])
                    count_run += 1
    AUC_SS = pd.DataFrame(data_pd, columns=['Int', 'Run', 'SleepState', 'LL'])
    return AUC_SS


# Define a function to calculate normalized AUC for a group
def calculate_normalized_auc(group, max_overall, max_Int):
    if len(group) < 10:
        # Not enough data points to calculate AUC
        return np.nan
    else:
        # Normalize LL

        LL_norm = (group['LL'] - np.min(group['LL'])) / (max_overall - np.min(group['LL']))
        # Normalize Int
        Int_norm = group['Int'] / max_Int
        # Calculate AUC
        return sklearn.metrics.auc(Int_norm, LL_norm)


def get_AUC_stats(sc, rc, con_trial, EEG_resp, cond_val):
    # for specfic connection sc- rc get AUC stats given by cond_val (SleepState_val)
    AUC_SS = get_AUC_trials(sc, rc, con_trial, EEG_resp, cond_val, n_trial=3, n_shuffle=10, w=0.25)
    # Calculate the overall max for normalization purposes
    max_overall = np.max(AUC_SS.groupby(['Int', 'SleepState'], as_index=False)['LL'].mean()['LL'])
    max_Int = np.max(AUC_SS['Int'])
    # get AUC distribution
    # Apply the function to each group and create a new DataFrame with the results
    AUC_results = AUC_SS.groupby(['SleepState', 'Run']).apply(calculate_normalized_auc, max_overall,
                                                              max_Int).reset_index()
    AUC_results.columns = ['SleepState', 'Run', 'AUC']

    # Compute mean and standard deviation of AUC for each SleepState
    auc_stats = AUC_results.groupby('SleepState')['AUC'].agg(['mean', 'std']).reset_index()

    # Rename columns
    auc_stats.columns = ['SleepState', 'AUC_mean', 'AUC_std']

    # Initialize columns for Cohen's d and p-value
    auc_stats['cohens_d'] = np.nan
    auc_stats['p_value'] = np.nan

    # Calculate Cohen's d and p-value for each SleepState compared to Wake
    wake_auc = AUC_results[AUC_results['SleepState'] == 'Wake']['AUC']

    for index, row in auc_stats.iterrows():
        if row['SleepState'] != 'Wake':
            current_state_auc = AUC_results[AUC_results['SleepState'] == row['SleepState']]['AUC']
            if (np.nanmean(current_state_auc) > 0) & (np.nanmean(wake_auc) > 0):
                # Calculate Cohen's d
                cohens_d = (row['AUC_mean'] - wake_auc.mean()) / np.sqrt(
                    (row['AUC_std'] ** 2 + wake_auc.std() ** 2) / 2)
                auc_stats.at[index, 'cohens_d'] = cohens_d
                # T-test
                t_stat, p_value = scipy.stats.ttest_ind(current_state_auc, wake_auc, nan_policy='omit')
                auc_stats.at[index, 'p_value_T'] = p_value
                t_stat, p_value = scipy.stats.mannwhitneyu(current_state_auc[~np.isnan(current_state_auc)].values,
                                                           wake_auc[~np.isnan(wake_auc)].values)
                auc_stats.at[index, 'p_value_MWU'] = p_value
                if row['AUC_mean'] < wake_auc.mean():
                    auc_stats.at[index, 'AUC_Ratio'] = row['AUC_mean'] / wake_auc.mean() - 1
                else:
                    auc_stats.at[index, 'AUC_Ratio'] = 1 - wake_auc.mean() / row['AUC_mean']
            else:
                auc_stats.at[index, 'AUC_Ratio'] = np.nan
                auc_stats.at[index, 'p_value_MWU'] = np.nan
                auc_stats.at[index, 'p_value_T'] = np.nan
                auc_stats.at[index, 'cohens_d'] = np.nan
    return auc_stats


def get_AUC_stats_across_connections(con_trial, auc_summary, EEG_resp, cond_val):
    stimchans = np.unique(con_trial['Stim']).astype(int)
    # respchans = np.array([69])  # np.unique(con_trial['Chan']).astype(int)
    auc_stats_across = pd.DataFrame()  # Initialize empty DataFrame to collect data

    for sc in stimchans:
        respchans = np.unique(
            auc_summary.loc[(auc_summary.Stim == sc) & (auc_summary.AUC_p < 0.05) & (auc_summary.MAX_p < 0.05), 'Chan'])
        for rc in respchans:
            if len(con_trial[(con_trial.Stim == sc) & (
                    con_trial.Chan == rc)]) > 0:  # Assuming you want to exclude self-connections
                auc_stats = get_AUC_stats(sc, rc, con_trial, EEG_resp, cond_val)
                if not auc_stats.empty:
                    auc_stats['Chan'] = rc
                    auc_stats['Stim'] = sc
                    auc_stats_across = pd.concat([auc_stats_across, auc_stats], ignore_index=True)

    return auc_stats_across
