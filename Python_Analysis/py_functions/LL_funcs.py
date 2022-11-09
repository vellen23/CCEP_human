import os
import numpy as np
import mne
import h5py
import scipy.fftpack
import matplotlib
import pywt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy import signal
import time
import seaborn as sns
import scipy.io as sio
from scipy.integrate import simps
import pandas as pd
from scipy import fft
import freq_funcs as ff
import sys
import basic_func as bf


def get_LL_bl(data, wdp_S, win_S, Fs):
    wdp = np.int(Fs * wdp_S)  # sliding window size in samples
    win = np.int(Fs * win_S)  # total window that is analyzed 1.5s
    EEG_pad = np.pad(data, [(0, 0), (0, 0), (np.int(wdp / 2), np.int(wdp / 2))], 'reflect')  # (18, 3006)
    LL_trial = np.zeros((data.shape[0], data.shape[1], win))
    for i in range(win):  # calculate LL for sliding window. take values of +- 1/2 wdp
        n = i + np.int(wdp / 2)
        LL_trial[:, :, i] = np.sum(abs(np.diff(EEG_pad[:, :, n - np.int(wdp / 2):n + np.int(wdp / 2)], axis=-1)),
                                   axis=-1)
    LL_mean = np.mean(LL_trial, axis=(1, 2))
    LL_std = np.std(LL_trial, axis=(1, 2))
    LL_thr = LL_mean + 2 * LL_std

    return LL_mean, LL_std, LL_thr, LL_trial


def get_P2P_resp(data, Fs, IPI, t_0):  # specific for Int and IPI
    # t_0 after how many seconds starts the stimulation --> -self.dur[0, 0]
    start = np.int((t_0 + IPI / 1000 + 0.01) * Fs)  # 50ms after second response
    end = np.int((t_0 + IPI / 1000 + 0.07) * Fs)  # 300ms after second response
    end2 = np.int((t_0 + IPI / 1000 + 0.04) * Fs)

    resp_pks = np.zeros((data.shape[0], data.shape[1], 3))
    resp_pks_loc = np.zeros((data.shape[0], data.shape[1], 2))
    resp_pks[:, :, 0] = np.min(data[:, :, start:end], axis=-1)
    resp_pks_loc[:, :, 0] = (np.argmin(data[:, :, start:end], axis=-1) + start - (t_0) * Fs) / Fs
    resp_pks[:, :, 1] = np.max(data[:, :, start:end2], axis=-1)
    resp_pks_loc[:, :, 1] = (np.argmax(data[:, :, start:end2], axis=-1) + start - (t_0) * Fs) / Fs
    resp_pks[:, :, 2] = resp_pks[:, :, 1] - resp_pks[:, :, 0]

    return resp_pks, resp_pks_loc


def get_RMS_resp(data, Fs, IPI, t_0, win):  # specific for Int and IPI
    # t_0 after how many seconds starts the stimulation --> -self.dur[0, 0]
    start = np.int((t_0 + IPI / 1000 + 0.012) * Fs)  # 15ms after second response
    end = np.int((t_0 + IPI / 1000 + 0.012 + win) * Fs)  # 515ms after second response

    resp_RMS = np.zeros((data.shape[0], data.shape[1], 1))
    resp_RMS[:, :, 0] = np.sqrt(np.mean(data[:, :, start:end] ** 2, axis=2))

    return resp_RMS


def get_LL_resp(data, Fs, IPI, t_0, win):  # specific for Int and IPI
    # t_0 after how many seconds starts the stimulation --> -self.dur[0, 0]
    # win in seconds
    start = np.int((t_0 + IPI / 1000 + 0.010) * Fs)  # 15 ms after second response
    end = np.int((t_0 + IPI / 1000 + 0.010 + win) * Fs)  # eg. win= 0.300 --> 315ms after second response

    resp_LL = np.zeros((data.shape[0], data.shape[1], 1))
    resp_LL[:, :, 0] = np.sum(abs(np.diff(data[:, :, start:end], axis=-1)), axis=-1) / (win * 1000)

    return resp_LL


def get_N1peaks_mean(sc, rc, LL_CCEP, EEG_resp):
    lists = LL_CCEP[
        (LL_CCEP['Artefact'] == 0) & (LL_CCEP['Int'] >= 6) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
    stimNum_all = lists.Num.values.astype('int')
    resp_all = bf.zscore_CCEP(ff.lp_filter(np.mean(EEG_resp[rc, stimNum_all, :], 0), 40, Fs))
    if np.max(abs(resp_all[Fs:int(1.5 * Fs)])) > 4:
        start_resp = 0
        peaks_all, properties_all = scipy.signal.find_peaks(abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]),
                                                            height=1.5, prominence=0.05, distance=0.03 * Fs, width=1)  #
        if len(peaks_all) > 0:
            peaks_all = peaks_all[0]
            w = \
                scipy.signal.peak_widths(abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]), [peaks_all],
                                         rel_height=0.5)[0]

            start_resp = (peaks_all - w) / Fs - 0.01

            if start_resp < 0.01:
                start_resp = 0

        pk, pk_s, p = get_peaks_all(resp_all, start_resp)

    else:
        pk_s = np.zeros((4, 1))
        p = 0

    return pk_s[0], p


def pk_lin_fit(resp, pk, fig=1, n_peaks=2, t_0=1, Fs=500):
    slope = np.zeros((n_peaks,))
    for i in range(n_peaks):
        if ~np.isnan(pk[i, 0]) & ~np.isnan(pk[i + 1, 0]):
            if pk[i, 1] < pk[i + 1, 1]:
                x = np.linspace(pk[i, 1] * Fs, pk[i + 1, 1] * Fs, int((pk[i + 1, 1] * Fs - pk[i, 1] * Fs)),
                                endpoint=False)
                # print(x+t_0*Fs)
                y = resp[(x + t_0 * Fs).astype('int')]
                x_s = x / Fs
                coef = np.polyfit(x, y, 1)
                slope[i] = abs(coef[0] / Fs * 1000)  # in uV/ms
                if fig:
                    poly1d_fn = np.poly1d(coef)
                    plt.plot(x_s, poly1d_fn(x), '--k')
    return slope


def get_LL_both(data, Fs, IPI, t_0=1, win=0.25):  # specific for Int and IPI
    # get LL of first and second pulse. put first pulse LL to nan if window is larger than IPI
    # t_0 = 5
    # w = 0.1
    art = 0.01
    resp_LL = np.zeros((data.shape[0], data.shape[1], 2))
    # first pulse
    # IPI_start = np.round((t_0+art)*Fs) # start position at first trigger plus 20ms (art removal), time zero
    w_start = np.int64(np.round((t_0 + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((t_0 + win) * Fs) - 1)
    n = np.int64((w_end - w_start))
    inds = np.linspace(w_start, w_end, n).T.astype(int)
    inds = np.expand_dims(inds, axis=(0, 1))
    # set to nan if IPI is smaller than LL window
    nans_ind = np.where(IPI < (win) * 1000)[0]
    resp_LL[:, :, 0] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)
    resp_LL[:, nans_ind, 0] = np.nan

    # second pulse
    w_start = np.int64(
        np.round((IPI / 1000 + t_0 + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((IPI / 1000 + t_0 + win) * Fs) - 1)
    n = np.int64((w_end - w_start)[0, 0])
    inds = np.linspace(w_start, w_end, n).T.astype(int)

    resp_LL[:, :, 1] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)
    return resp_LL  # chns x stims x SP/PP LL


def get_peaks_trial(resp, peak_req, p=1, t_0=1, Fs=500):
    # p = polarity, how to change that N1 and N2 are local maxima
    # peak_req, requirement where peak N1, P1 are 2x array

    x = p * bf.zscore_CCEP(resp)
    x1 = int((t_0) * Fs)
    x2 = int((t_0 + 1.5) * Fs)
    y = x[x1:x2]
    pk = np.zeros((4, 2))

    # first peaks (N1, P1)
    i = 0
    for k in [1, -1]:
        # 1. find all peaks
        peaks_all, properties_all = scipy.signal.find_peaks(k * y, prominence=0.5, distance=0.03 * Fs)  #
        # 2. only get peaks with highest prominences
        # thr                        = 0.5*(properties["prominences"][np.argsort(properties["prominences"])[-np.min([len(peaks_all),3])]])
        # peaks_all, properties_all  = scipy.signal.find_peaks(k*y, prominence=thr, width=1, distance=0.05*Fs)#

        if i == 0:  # N1
            w = 0.05 * Fs
            # w   = 0.3*Fs
        else:  # P1
            # w   = abs(peak_req[1]-peak_req[0])*Fs
            w = 0.1 * Fs

        # peak 1
        if len(peaks_all) > 0:
            req = (peaks_all > pk[0, 0] * Fs) & (peaks_all > 0.012 * Fs) & (peaks_all > peak_req[i] * Fs - w) & (
                    peaks_all < peak_req[i] * Fs + w)
            if np.mean(req) == 0:
                pk[i, 0] = np.nan
                pk[i, 1] = np.nan
            else:
                peak = peaks_all[req]
                properties = properties_all
                for item in properties.items():
                    properties[item[0]] = properties[item[0]][req]
                if len(peak) > 1:
                    ix = np.argsort(properties["prominences"])[-1]
                    peak = peak[ix]
                    for item in properties.items():
                        properties[item[0]] = properties[item[0]][ix]

                pk[i, 0] = peak / Fs
                pk[i, 1] = resp[peak + x1]
        else:
            pk[i, 0] = np.nan
            pk[i, 1] = np.nan
        i = i + 1

    # peak 2, N2/P2
    i = 0
    for k in [1, -1]:
        # 1. find all peaks
        if i == 0:  # N2
            y_filt = ff.lp_filter(y, 20, Fs)
            pro = 0.5
            # w           = 1.5 *abs(peak_req[i+2]-peak_req[i+1])
            w = 0.05 * Fs
        else:
            y_filt = ff.lp_filter(y, 5, Fs)
            # w           = 2 *abs(peak_req[i+2]-peak_req[i+1])
            w = 0.05 * Fs
            pro = 0.5
        peaks_all, properties_all = scipy.signal.find_peaks(k * y_filt, prominence=pro, distance=0.05 * Fs)  #
        # 2. only get peaks with highest prominences
        # thr    = (properties["prominences"][np.argsort(properties["prominences"])[-np.min([len(peaks_all),3])]])
        # peaks_all, properties_all     = scipy.signal.find_peaks(k*y, prominence=thr, width=1, distance=0.05*Fs)#
        if len(peaks_all) > 0:
            req = np.zeros((len(peaks_all),))
            while np.mean(req) == 0:  # not req
                # req   = (peaks_all>pk[1+i,0]*Fs+0.02*Fs)&(peaks_all<pk[1+i,0]*Fs+w)
                req = (peaks_all > pk[1 + i, 0] * Fs + 0.02 * Fs) & (peaks_all > peak_req[2 + i] * Fs - w) & (
                        peaks_all < peak_req[2 + i] * Fs + w)
                w = w + 0.05 * Fs
                if (w > Fs) & (np.mean(req) == 0):
                    req = -1
            if np.mean(req) == -1:
                pk[i + 2, 0] = np.nan
                pk[i + 2, 1] = np.nan
            else:

                peak = peaks_all[req]
                properties = properties_all
                for item in properties.items():
                    properties[item[0]] = properties[item[0]][req]
                if len(peak) > 1:
                    ix = np.argsort(properties["prominences"])[-1]
                    peak = peak[np.argsort(properties["prominences"])[-1]]
                    for item in properties.items():
                        properties[item[0]] = properties[item[0]][ix]
                if abs(resp[peak + x1]) > 2:
                    pk[i + 2, 0] = peak / Fs
                    pk[i + 2, 1] = resp[peak + x1]
                else:
                    pk[i + 2, 0] = np.nan
                    pk[i + 2, 1] = np.nan
                # print(str(k)+' -- ' + str(properties["prominences"]))
        else:
            pk[i + 2, 0] = np.nan
            pk[i + 2, 1] = np.nan

        i = i + 1

    return pk


def get_peaks_all(resp_all, start_resp, t_0=1, Fs=500):
    # intended for mean responses, where polarity of N-peaks is unknown

    pk_all = np.zeros((6, 2))
    i = 0
    pro_min = -10
    for k in [-1, 1]:
        # selected highest peaks
        pk, properties_all = scipy.signal.find_peaks(k * resp_all, height=1, prominence=0.002, distance=0.03 * Fs,
                                                     width=2)
        if len(pk) > 0:
            pro_min = np.max([pro_min, np.sort(properties_all['prominences'])[-np.min([len(pk), 5])]])

    for k in [-1, 1]:
        # selected highest peaks
        pk, properties_all = scipy.signal.find_peaks(k * resp_all, height=1, prominence=pro_min, distance=0.03 * Fs,
                                                     width=2)  #
        if len(pk) > 0:
            # print(properties_all)
            req_N1 = (pk > (t_0 + start_resp + 0.001) * Fs) & (pk < (t_0 + start_resp + 0.06) * Fs)
            req_N2 = (pk > (t_0 + start_resp + 0.08) * Fs) & (pk < (t_0 + start_resp + 1) * Fs)

            if any(req_N1) & any(req_N2):
                j = 0
                for req in [req_N1, req_N2]:
                    pk, properties = scipy.signal.find_peaks(k * resp_all, height=1, prominence=pro_min,
                                                             distance=0.03 * Fs,
                                                             width=1)  #
                    pk_N = pk[req]
                    for item in properties.items():
                        properties[item[0]] = properties[item[0]][req]
                    if len(pk_N) > 1:
                        ix = np.argsort(properties["prominences"])[-1]
                        pk_N = np.array([pk_N[ix]])
                        for item in properties.items():
                            properties[item[0]] = properties[item[0]][ix]

                    pk_all[j, i] = pk_N
                    pk_all[4, i] = pk_all[4, i] + properties['prominences']
                    if j == 2:
                        # print(scipy.signal.peak_widths(k*resp_all, pk_N, rel_height=1))
                        pk_all[5, i] = scipy.signal.peak_widths(k * resp_all, pk_N, rel_height=1)[0]
                        # pk_all[5,i] = properties['widths']
                        # peak_widths(x, peaks, rel_height=1)

                    j = 2
                ## P1
                pk, properties = scipy.signal.find_peaks(-k * resp_all, prominence=0.005, distance=0.03 * Fs,
                                                         width=1)  #
                pro_min = np.sort(properties['prominences'])[-np.min([len(pk), 15])]
                pk, properties = scipy.signal.find_peaks(-k * resp_all, prominence=pro_min, distance=0.015 * Fs,
                                                         width=1)  #
                req_P1 = (pk > pk_all[0, i] + 0.008 * Fs) & (pk < pk_all[2, i] - 0.008 * Fs)  # between N1 and N2
                pk_P = pk[req_P1]
                for item in properties.items():
                    properties[item[0]] = properties[item[0]][req_P1]
                if len(pk_P) > 0:
                    ix = np.argsort(properties["prominences"])[-1]
                    pk_P = np.array([pk_P[ix]])
                    pk_all[1, i] = pk_P

                ## P 2
                resp_fil = ff.lp_filter(resp_all, 15, Fs)
                pk, properties = scipy.signal.find_peaks(-k * resp_fil, prominence=0.005, distance=0.03 * Fs,
                                                         width=1)  #
                req_P1 = (pk > pk_all[2, i] + 0.015 * Fs) & (pk < pk_all[2, i] + 1 * Fs)
                pk_P = pk[req_P1]
                for item in properties.items():
                    properties[item[0]] = properties[item[0]][req_P1]
                if len(pk_P) > 0:
                    ix = np.argsort(properties["prominences"])[-1]
                    pk_P = np.array([pk_P[ix]])
                    pk_all[3, i] = pk_P


            else:
                pk_all[4, i] = 0
                pk_all[5, i] = 10000
            # print(str(k)+' - polarity does not fulfill N peak requirement')
        else:
            pk_all[4, i] = 0
            pk_all[5, i] = 10000
        i = i + 1
    pk_all = pk_all.astype('int')
    if pk_all[4, 1] < pk_all[4, 0]:
        pk = pk_all[0:4, 0]
        p = -1
    elif pk_all[4, 1] > pk_all[4, 0]:
        pk = pk_all[0:4, 1]
        p = 1
    else:
        pk = np.ones((4,)) * 500
        p = 0
    peak_s = (pk - t_0 * Fs) / Fs
    return pk.astype('int'), peak_s, p


def get_peaks_trials(resp, peak_req, t_0=1, Fs=500):
    # p = polarity, how to change that N1 and N2 are local maxima
    # peak_req, requirement where peak N1, P1 are 2x array
    p = peak_req[1]
    N1_onset = int((peak_req[0] + t_0) * Fs)
    N1_w = np.zeros((2,)).astype('int')
    w_N1 = int(0.01 * Fs)
    N1_w[0] = np.min([t_0 * Fs, N1_onset - w_N1]).astype('int')
    N1_w[1] = np.max([N1_w[0] + 0.08 * Fs, N1_onset + w_N1]).astype('int')

    resp = p * resp
    pk = np.zeros((len(resp), 3, 2))

    pk[:, 0, 0] = np.max(resp[:, N1_w[0]:N1_w[1]], -1)
    pk[:, 0, 1] = np.argmax(resp[:, N1_w[0]:N1_w[1]], -1) + N1_w[0]
    # N2
    for i in range(len(resp)):
        peaks_all, properties = scipy.signal.find_peaks(
            ff.lp_filter(resp[i, int(pk[i, 0, 1]):(pk[i, 0, 1] +1 * Fs).astype('int')], 15, Fs), prominence=0.01,
            distance=1,
            width=0.01 * Fs)  #
        if len(peaks_all) > 0:
            pk[i, 2, 1] = int(peaks_all[(np.argsort(properties["prominences"])[-1])] + int(pk[i, 0, 1]))
            pk[i, 2, 0] = resp[i, int(pk[i, 2, 1])]
            pk[i, 1, 1] = np.argmin(resp[i, int(pk[i, 0, 1]):int(pk[i, 2, 1])]) + int(pk[i, 0, 1])
            pk[i, 1, 0] = np.min(resp[i, int(pk[i, 0, 1]):int(pk[i, 2, 1])])

        else:
            pk[i, 2, 1] = np.nan

    pk[:, :, 0] = pk[:, :, 0] * p
    return pk



def pk2pk(data, Fs, t_0):
    start = np.int64((t_0 + 0.015) * Fs)  # 50ms after second response
    end = np.int64((t_0 + 0.10) * Fs)  # 50ms after second response
    resp_pks = np.zeros((data.shape[0], 3))
    resp_pks_loc = np.zeros((data.shape[0], 2))
    resp_pks[:, 0] = np.min(data[:, start:end], axis=-1)
    resp_pks[:, 1] = np.max(data[:, start:end], axis=-1)
    resp_pks[:, 2] = resp_pks[:, 1] - resp_pks[:, 0]
    resp_pks_loc[:, 1] = (np.argmax(data[:, start:end], axis=-1) + start - (t_0) * Fs) / Fs
    resp_pks_loc[:, 0] = (np.argmin(data[:, start:end], axis=-1) + start - (t_0) * Fs) / Fs
    return resp_pks, resp_pks_loc


def get_LL_full(data, Fs, IPI, t_0=5, win=0.1):
    art = 0.01
    resp_LL = np.zeros((data.shape[0], data.shape[1], 2))
    # first pulse
    # IPI_start = np.round((t_0+art)*Fs) # start position at first trigger plus 20ms (art removal), time zero
    w_start = np.int64(np.round((t_0 + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((t_0 + win) * Fs) - 1)
    n = np.int64((w_end - w_start))
    inds = np.linspace(w_start, w_end, n).T.astype(int)
    inds = np.expand_dims(inds, axis=(0, 1))
    # set to nan if IPI is smaller than LL window
    nans_ind = np.where(IPI < (win) * 1000)[0]
    resp_LL[:, :, 0] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)
    resp_LL[:, nans_ind, 0] = np.nan

    # second pulse
    w_start = np.int64(
        np.round((IPI / 1000 + t_0 + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((IPI / 1000 + t_0 + win) * Fs) - 1)
    n = np.int64((w_end - w_start)[0, 0])
    inds = np.linspace(w_start, w_end, n).T.astype(int)

    resp_LL[:, :, 1] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)
    return resp_LL  # chns x stims x SP/PP LL


def get_LL_ratio(data, Fs, IPI, t_bl=0.8, t_stim=1, win=0.15):
    # LL of timepoint during BL, l of REsp. get RAtio
    # t_0 after how many seconds starts the stimulation --> -self.dur[0, 0]
    art = 0.015
    # win in seconds, BL not depeding on IPI
    w_start = np.int64(np.round((t_bl + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((t_bl + win) * Fs) - 1)
    n = np.int64((w_end - w_start))
    inds = np.linspace(w_start, w_end, n).T.astype(int)
    inds = np.expand_dims(inds, axis=(0, 1))
    LL_bl = np.zeros((data.shape[0], data.shape[1], 1))
    LL_bl[:, :, 0] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)

    # Stimulation (including IPI)
    w_start = np.int64(
        np.round((IPI / 1000 + t_stim + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((IPI / 1000 + t_stim + win) * Fs) - 1)
    n = np.int64((w_end - w_start)[0, 0])
    inds = np.linspace(w_start, w_end, n).T.astype(int)

    LL_resp = np.zeros((data.shape[0], data.shape[1], 3))
    LL_resp[:, :, 0] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)
    LL_resp[:, :, 1] = LL_resp[:, :, 0] / LL_bl[:, :, 0]  # LL ratio compared to BL
    LL_resp[:, :, 2] = LL_bl[:, :, 0]  # BL absolute LL
    return LL_resp


def get_HGP_both(data, Fs, IPI, t_0=5, win=0.3):  # specific for Int and IPI
    art = 0.015
    max_HGP = np.zeros((data.shape[0], data.shape[1], 2))
    # first pulse
    w_start = np.round((t_0 + art) * Fs)  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.round((t_0 + win) * Fs) - 1
    n = np.int64((w_end - w_start))
    inds = np.linspace(w_start, w_end, n).T.astype(int)
    inds = np.expand_dims(inds, axis=(0, 1))
    nans_ind = np.where(IPI < (win) * 1000)[0]
    max_HGP[:, :, 0] = np.max(np.take_along_axis(data, inds, axis=2), -1)
    max_HGP[:, nans_ind, 0] = np.nan

    # second pulse
    w_start = np.round((IPI / 1000 + t_0 + art) * Fs)  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.round((IPI / 1000 + t_0 + win) * Fs) - 1
    n = np.int64((w_end - w_start)[0, 0])
    inds = np.linspace(w_start, w_end, n).T.astype(int)
    max_HGP[:, :, 1] = np.max(np.take_along_axis(data, inds, axis=2), -1)
    return max_HGP  # chns x stims x SP/PP LL


def get_NNMFcoeff_both(data, Fs, stim_list, t_0=1, win=0.1):  # specific for Int and IPI
    start = 4  # 4s
    # get LL of first and second pulse. put frist pulse LL to nan if window is larger than IPI
    # t_0 = 5
    # w = 0.1
    art = 0.015
    resp_coeff = np.zeros((data.shape[0], len(stim_list), 2))  # number of channels,
    for s in range(len(stim_list)):
        IPI = stim_list.IPI_ms.values[s]
        if IPI > win * 1000:
            resp_coeff[:, s, 0] = np.nanmean(
                data[:, np.int(((s + 1) * 4 + t_0 + art) * Fs):np.int(((s + 1) * 4 + t_0 + win) * Fs)], 1)
        else:
            resp_coeff[:, s, 0] = np.nan

        resp_coeff[:, s, 1] = np.nanmean(data[:, np.int(((s + 1) * 4 + t_0 + art + IPI / 1000) * Fs):np.int(
            ((s + 1) * 4 + t_0 + win + IPI / 1000) * Fs)], 1)

    return resp_coeff  # chns x stims x SP/PP LL


def get_LL_all(data, Fs, win):  # specific for Int and IPI
    ## LL for entire signal
    wdp = np.int64(Fs * win)  # 100ms -> 50 sample points
    EEG_pad = np.pad(data, [(0, 0), (0, 0), (np.int64(wdp / 2), np.int64(wdp / 2))], 'constant',
                     constant_values=(0, 0))  # 'reflect'(18, 3006)
    LL_trial = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    # LL_max = np.zeros((data.shape[0], data.shape[1], 2))
    for i in range(data.shape[2]):  # entire response
        n = i + np.int64(wdp / 2)
        LL_trial[:, :, i] = np.nansum(abs(np.diff(EEG_pad[:, :, n - np.int64(wdp / 2):n + np.int64(wdp / 2)], axis=-1)),
                                      axis=-1) / (win * 1000)
    # onset in ms after probing pulse, must be at least 10ms + half of the sliding window to not include fake data
    return LL_trial


# def get_LL_all(data, Fs,win, t0, IPI):  # specific for Int and IPI
#     # if w is 100ms - than the LL_max can be taken until 250. (300ms- 1/2win)
#     t_max               = 0.3 - 0.5*win
#     t1                  = np.int(t0 * Fs) #start conditioning stim, time zero
#     t2                  = np.int((t0 + IPI / 1000) * Fs)#start probing stim
#     #blank out stim artifact
#     #data[:, :, np.int(t1 - 0.002 * Fs):np.int(t1 + 0.01 * Fs)] = 0
#     #data[:, :, np.int(t2 - 0.002 * Fs):np.int(t2 + 0.01 * Fs)] = 0
#     #start_samp  = np.int(((-dur[0, 0]) + IPI / 1000 + 0.02) * Fs)  # 20ms after second response
#     #wdp         = np.int(Fs * wdp_S)  # sliding window size in samples
#     wdp         = np.int(Fs * win)  # 100ms -> 50 sample points
#     EEG_pad     = np.pad(data, [(0, 0), (0, 0), (np.int(wdp / 2), np.int(wdp / 2))], 'reflect')  # (18, 3006)
#     LL_trial    = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
#     LL_max      = np.zeros((data.shape[0], data.shape[1], 2))
#     for i in range(data.shape[2]):  # entire response
#         n                       = i + np.int(wdp / 2)
#         LL_trial[:, :, i]       = np.nansum(abs(np.diff(EEG_pad[:, :, n - np.int(wdp / 2):n + np.int(wdp / 2)], axis=-1)),
#                                    axis=-1)/(win*1000)
#     LL_max[:, :, 0]         = np.nanmax(LL_trial[:,:,np.int(t2 + (0.01+win/2) * Fs):np.int(t2 + t_max * Fs)], axis=2)
#     LL_max[:, :, 1]         = (np.argmax(LL_trial[:,:,np.int(t2 + (0.01+win/2) * Fs):np.int(t2 + t_max * Fs)], axis=2)+np.int((0.01+win/2) * Fs))/Fs*1000
#     # onset in ms after probing pulse, must be at least 10ms + half of the sliding window to not include fake data
#     return LL_trial, LL_max #, LL_pwr #, onset

def get_LL_all_old(data, Fs, win, t0, IPI):  # specific for Int and IPI
    t1 = np.int(t0 * Fs)
    t2 = np.int((t0 + IPI / 1000) * Fs)
    data[:, :, t1 - 0.002 * Fs:t1 + 0.01 * Fs] = np.nan
    data[:, :, t2 - 0.002 * Fs:t2 + 0.01 * Fs] = np.nan
    # start_samp  = np.int(((-dur[0, 0]) + IPI / 1000 + 0.02) * Fs)  # 20ms after second response
    # wdp         = np.int(Fs * wdp_S)  # sliding window size in samples
    wdp = np.int(Fs * win)  # 480ms
    EEG_pad = np.pad(data, [(0, 0), (0, 0), (np.int(wdp / 2), np.int(wdp / 2))], 'reflect')  # (18, 3006)
    LL_trial = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    # LL_pwr      = np.zeros((data.shape[0], data.shape[1], 1))  # LL of 500ms of response
    onset = np.zeros((data.shape[0], data.shape[1], 1))
    for i in range(data.shape[2]):  # entire response
        n = i + np.int(wdp / 2)
        LL_trial[:, :, i] = np.sum(abs(np.diff(EEG_pad[:, :, n - np.int(wdp / 2):n + np.int(wdp / 2)], axis=-1)),
                                   axis=-1)
    # LL_pwr[:, :, 0] = np.sum(abs(np.diff(EEG_pad[:, :, start_samp:start_samp + win], axis=-1)), axis=-1)

    # # todo: find onset
    # try:
    #     onset[:, :, 0] = (list(map(lambda i: i > thr[:], LL_trial[:, :, start_samp])).index(
    #         True) + np.int(0.02 * Fs)) / Fs * 1000  # when does LL array crosses threshold (mean+2std)
    # except ValueError:
    #     onset[:, :, 0] = 20  # in ms

    return LL_trial  # , LL_pwr #, onset


def get_WT(data, Fs):
    T_samp = 1 / Fs
    scale = np.geomspace(0.5, 200, 100) * 5  # f from 0.5 - 200 in 100 steps

    # calculate pwr for each trial
    coef, freqs = pywt.cwt(data, scale, "cmor1.5-1.0", sampling_period=T_samp)
    pwr = abs(coef)  # absolute values
    phase = np.angle(coef)
    phase_mean = np.nanmean(phase, axis=2)  # mean for all trials
    pwr_mean = np.nanmean(pwr, axis=2)
    return pwr, pwr_mean, freqs, phase, phase_mean


def get_HT(data, Fs):
    y = fft(data)
    n = data.shape[2]
    y = y[:, :, 0:np.int(n / 2)]
    f = np.arange(0, n / 2) * Fs / n  # (0:n/2-1)*(Fs/n)
    power = abs(y) ** 2 / n
    power_mean = np.nanmean(power, axis=1)
    return f, power, power_mean


def get_FFT(data, Fs):
    y = fft(data)
    n = data.shape[2]
    y = y[:, :, 0:np.int(n / 2)]
    f = np.arange(0, n / 2) * Fs / n  # (0:n/2-1)*(Fs/n)
    power = abs(y) ** 2 / n
    power_mean = np.nanmean(power, axis=1)
    return f, power, power_mean


def get_RMS(data):
    rms = np.sqrt(np.mean(data ** 2, axis=1))
    return rms


def get_HGP_both(data, Fs, IPI, t_0=1, win=0.3):  # specific for Int and IPI
    art = 0.015
    max_HGP = np.zeros((data.shape[0], data.shape[1], 2))
    # first pulse
    w_start = np.round((t_0 + art) * Fs)  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.round((t_0 + win) * Fs) - 1
    n = np.int64((w_end - w_start))
    inds = np.linspace(w_start, w_end, n).T.astype(int)
    inds = np.expand_dims(inds, axis=(0, 1))
    nans_ind = np.where(IPI < (win) * 1000)[0]
    max_HGP[:, :, 0] = np.max(np.take_along_axis(data, inds, axis=2), -1)
    max_HGP[:, nans_ind, 0] = np.nan

    # second pulse
    w_start = np.round((IPI / 1000 + t_0 + art) * Fs)  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.round((IPI / 1000 + t_0 + win) * Fs) - 1
    n = np.int64((w_end - w_start)[0, 0])
    inds = np.linspace(w_start, w_end, n).T.astype(int)
    max_HGP[:, :, 1] = np.max(np.take_along_axis(data, inds, axis=2), -1)
    return max_HGP  # chns x stims x SP/PP LL
