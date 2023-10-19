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


def cal_delay(signal, WOI=0, t_0=1, Fs=500, w_LL_onset=0.05, plot=0):
    # Response onset: peak of second derivative
    # 1. first get LL of response
    # .2. peak of second derivative to get response osnet
    import matplotlib.pyplot as plt
    smooth_win = int(w_LL_onset * Fs)
    if np.mod(smooth_win, 2) == 0:
        smooth_win = smooth_win + 1

    LL_transform = LLf.get_LL_all(np.expand_dims(ff.lp_filter(signal, 30, Fs), [0, 1]), Fs, w_LL_onset)[
        0, 0]

    # Smooth LL data
    d1_LL = scipy.signal.savgol_filter(LL_transform, smooth_win, 3, 1)  # First derivative
    d2_LL = scipy.signal.savgol_filter(LL_transform, smooth_win, 3, 2)  # Second derivative
    # d3_LL = scipy.signal.savgol_filter(LL_transform, smooth_win, 3, 3)  # Third derivative

    d2_LL[d1_LL < 0] = np.nan  # only increase intresting
    d2_LL[:int((
                           t_0 - w_LL_onset / 2 - 0.02) * Fs)] = np.nan  # not before Stimd2_LL[np.argmax(LL_transform[int((t_0 - w_LL_onset / 2) * Fs):int((t_0 - w_LL_onset / 2+0.5) * Fs)])+int((t_0 - w_LL_onset / 2)*Fs):] = np.nan  # not after LL peak (must bbe before)
    d2_LL[LL_transform < np.nanpercentile(LL_transform[int((t_0 - 0.1) * Fs):int((t_0 - 0.05) * Fs)],
                                          50)] = np.nan  # increae in LL
    arr = np.array(abs(signal) > 4 * np.nanpercentile(abs(signal[int((t_0 - 0.1) * Fs):int((t_0 - 0.05) * Fs)]),
                                                      99)) * 1  # before Signal is to large
    arr[:int((t_0 - 0.02) * Fs)] = 0  # arr = arr[arr >= 0]
    t_pk = sf.search_sequence_numpy(arr, np.ones((int((0.005) * Fs),)))
    if len(t_pk) > 0:
        d2_LL[int(t_pk[0] - Fs * (w_LL_onset / 2)):] = np.nan
    peaks_max, _ = find_peaks(d2_LL, height=0)
    if np.isnan(np.nanmax(d2_LL)):
        t_onset = WOI
    else:
        if len(peaks_max) > 0:
            t_onset = peaks_max[0]
        else:
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
    return t_onset
