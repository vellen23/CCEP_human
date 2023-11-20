import numpy as np

import h5py
import scipy.fftpack
from scipy.signal import find_peaks
import scipy.io as sio
import freq_funcs as ff
import LL_funcs as LLf
import significance_funcs as sf

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]


####GENERAL functions to analyze characteristics of a CCEP
def CCEP_metric(trials, t0=1, w_AUC=1, Fs=500):
    """
    Calculate CCEP metrics including N1, N2, and AUC.

    Parameters:
    trials (array): The trial data (trial x time).
    t0 (float): The time of stimulation onset in seconds.
    w_AUC (float): The width of the AUC window in seconds.
    Fs (int): The sampling frequency.

    Returns:
    tuple: N1, N2, AUC and P2P values.
    """
    # Mean response across trials
    mean = np.nanmean(trials, 0)
    mean = zscore_CCEP(mean)
    pol = -1 if abs(np.min(mean[int((t0 + 0.01) * Fs):int((t0 + 0.05) * Fs)])) > np.max(
            mean[int((t0 + 0.01) * Fs):int((t0 + 0.05) * Fs)]) else 1
    # if there is a neagitve peak within 50 ms -> pol = -1
    # if there is a postive peak within 50ms -> pol +
    #if there is a peak in both polarity, take the stronger one
    # if there is no peak,

    # Calculate N1 and N2
    N1 = np.max(pol * trials[:, int((t0 + 0.01) * Fs):int((t0 + 0.05) * Fs)], axis=1)
    N2 = np.max(pol * trials[:, int((t0 + 0.05) * Fs):int((t0 + 0.4) * Fs)], axis=1)
    # Get P2P within 1s
    peak_min = np.min(trials[:, int((t0 + 0.01) * Fs):int((t0 + w_AUC) * Fs)], axis=1)
    peak_max = np.max(trials[:, int((t0 + 0.01) * Fs):int((t0 + w_AUC) * Fs)], axis=1)
    P2P = abs(peak_max-peak_min)
    # Z-score the trials
    trials_z = zscore_CCEP(trials, t_0=t0, w0=0.5, Fs=Fs)

    # Calculate AUC
    auc_start = int(t0 * Fs)
    auc_end = int((t0 + w_AUC) * Fs)
    AUC = np.sum(np.abs(trials_z[:, auc_start:auc_end]), axis=1)

    return N1, N2, AUC, P2P

def zscore_CCEP(data, t_0=1, w0=0.5, Fs=500):
    if len(data.shape) == 1:
        m = np.mean(data[int((t_0 - w0) * Fs):int((t_0 - 0.05) * Fs)])
        s = np.std(data[int((t_0 - w0) * Fs):int((t_0 - 0.05) * Fs)])
        data = (data - m) / s
    else:
        m = np.nanmean(data[:, int((t_0 - w0) * Fs):int((t_0 - 0.05) * Fs)], -1)
        s = np.nanstd(data[:, int((t_0 - w0) * Fs):int((t_0 - 0.05) * Fs)], -1)
        m = m[:, np.newaxis]
        s = s[:, np.newaxis]
        data = (data - m) / s
    return data


def CCEP_onset(signal, WOI=0, t_0=1, Fs=500, w_LL_onset=0.05):
    """
    Calculate the onset of a Cortico-Cortical Evoked Potential (CCEP) in a signal.

    Parameters:
    - signal (array): Mean signal of one connection.
    - WOI (float): Onset of Window Of Interest based on previous LL calculations (connection-specific).
    - t_0 (float): Time of stimulation in the signal (e.g., for epoch: [-1, 3] -> t_0 = 1).
    - Fs (int): Sampling frequency.
    - w_LL_onset (float): Window length for onset detection.

    Returns:
    - float: Time of response onset after stimulation, in seconds.
    """

    # Calculate smoothing window length
    smooth_win = int(w_LL_onset * Fs)
    if np.mod(smooth_win, 2) == 0:
        smooth_win += 1

    # Filter the signal
    data_CCEP = ff.lp_filter(signal, 30, Fs)

    # Get LL transformation of the filtered signal
    LL_transform = LLf.get_LL_all(np.expand_dims(data_CCEP, [0, 1]), Fs, w_LL_onset)[0, 0]

    # Find the peak CCEP location
    start_idx = int(t_0 * Fs)
    end_idx = int((t_0 + WOI + 0.125) * Fs)
    pk_CCEP_loc = np.argmax(abs(data_CCEP[start_idx:end_idx]))

    # Smooth LL data to calculate first and second derivatives
    d1_LL = scipy.signal.savgol_filter(LL_transform, smooth_win, 3, 1)  # First derivative
    d2_LL = scipy.signal.savgol_filter(LL_transform, smooth_win, 3, 2)  # Second derivative

    # Apply constraints to second derivative data
    d2_LL[d1_LL < 0] = np.nan  # Ignore decreasing LL values
    d2_LL[:int((t_0 - w_LL_onset / 2 - 0.02) * Fs)] = np.nan  # Ignore values before stimulation
    d2_LL[int((t_0 - w_LL_onset / 2) * Fs + pk_CCEP_loc):] = np.nan  # Ignore values after CCEP peak

    # Find the peak in the second derivative, which indicates the strongest acceleration (response onset)
    t_onset = np.nanargmax(d2_LL) / Fs - t_0
    t_onset += w_LL_onset / 2  # Realign to account for window offset

    # Ensure onset time is not negative
    if t_onset < 0:
        t_onset = 0

    return t_onset

def cal_delay(signal, WOI=0, t_0=1, Fs=500, w_LL_onset=0.05, plot=0):
    # Response onset: peak of second derivative
    # 1. first get LL of response
    # 2. peak of second derivative to get response onset
    import matplotlib.pyplot as plt
    smooth_win = int(w_LL_onset * Fs)
    if np.mod(smooth_win, 2) == 0:
        smooth_win = smooth_win + 1
    data_CCEP = ff.lp_filter(signal, 30, Fs)
    LL_transform = LLf.get_LL_all(np.expand_dims(data_CCEP, [0, 1]), Fs, w_LL_onset)[
        0, 0]
    # pk_CCEP_loc = np.argmax(abs(ff.lp_filter(signal, 30, Fs)[t_0 * Fs:]))
    pk_CCEP_loc = np.argmax(abs(data_CCEP[int(t_0 * Fs):int((t_0 + WOI + 0.125) * Fs)]))  # onset is before peak in WOI
    # Smooth LL data
    d1_LL = scipy.signal.savgol_filter(LL_transform, smooth_win, 3, 1)  # First derivative
    d2_LL = scipy.signal.savgol_filter(LL_transform, smooth_win, 3, 2)  # Second derivative
    d2_LL0 = scipy.signal.savgol_filter(LL_transform, smooth_win, 3, 2)  # Second derivative

    d2_LL[d1_LL < 0] = np.nan  # only increase intresting
    d2_LL[:int((
                           t_0 - w_LL_onset / 2 - 0.02) * Fs)] = np.nan  # not before Stimd2_LL[np.argmax(LL_transform[int((t_0 - w_LL_onset / 2) * Fs):int((t_0 - w_LL_onset / 2+0.5) * Fs)])+int((t_0 - w_LL_onset / 2)*Fs):] = np.nan  # not after LL peak (must bbe before)
    d2_LL[int((t_0 * Fs) + pk_CCEP_loc):] = np.nan  # not before Stim
    # d2_LL[LL_transform < np.nanpercentile(LL_transform[int((t_0 - 0.1) * Fs):int((t_0 - 0.05) * Fs)],
    #                                       50)] = np.nan  # increae in LL
    # arr = np.array(abs(signal) > 4 * np.nanpercentile(abs(signal[int((t_0 - 0.1) * Fs):int((t_0 - 0.05) * Fs)]),
    #                                                   99)) * 1  # before Signal is to large
    # arr[:int((t_0 - 0.02) * Fs)] = 0  # arr = arr[arr >= 0]
    # t_pk = sf.search_sequence_numpy(arr, np.ones((int((0.005) * Fs),)))
    # if len(t_pk) > 0:
    #     d2_LL[int(t_pk[0] - Fs * (w_LL_onset / 2)):] = np.nan
    # peaks_max, _ = find_peaks(d2_LL, height=0)
    # if np.isnan(np.nanmax(d2_LL)):
    #     t_onset = WOI
    # else:
    #     if len(peaks_max) > 0:
    #         t_onset = peaks_max[0]
    #     else:
    #         t_onset = np.nanargmax(d2_LL)
    #     t_onset = t_onset / Fs - t_0  # - w_LL_onset / 3
    #     t_onset = t_onset + w_LL_onset / 2  # realign
    #     if t_onset < 0:
    #         t_onset = 0
    t_onset = np.nanargmax(d2_LL)
    t_onset = t_onset / Fs - t_0  # - w_LL_onset / 3
    t_onset = t_onset + w_LL_onset / 2  # realign
    if t_onset < 0:
        t_onset = 0
    if plot:
        x_ax = np.linspace(-1, 3, len(signal))
        plt.plot(x_ax, ff.lp_filter(signal, 45, Fs), color=[0, 0, 0])
        plt.axvline(0, color=[1, 0, 0])
        plt.axvline(t_onset, color=[0, 0, 0], ls='--')
        plt.xlim([-0.2, 0.3])
        plt.axhline(np.nanpercentile(LL_transform[int((t_0 - 0.2) * Fs):int((t_0 - 0.05) * Fs)], 95) * 100)
        # plt.axhline(5*np.nanpercentile(abs(signal[int((t_0 - 0.2) * Fs):int((t_0 - 0.05) * Fs)]), 99))
        plt.plot(x_ax + w_LL_onset / 2, LL_transform * 100, color=[0, 0, 0], alpha=0.7)
        plt.plot(x_ax + w_LL_onset / 2, d2_LL * 10000, color=[1, 0.3, 0])
        plt.show()
    return t_onset, LL_transform, d1_LL, d2_LL0
