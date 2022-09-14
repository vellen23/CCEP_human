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

######OLD Versions

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


def get_peaks(resp_all, t_0=1, Fs=500):
    # version 1, unknown polarity, find strongest peak
    x = resp_all
    x1 = int((t_0) * Fs)
    x2 = int((t_0 + 1) * Fs)
    y = resp_all[x1:x2]
    resp_all[x1: int((t_0 + 0.010) * Fs)] = np.nan
    pk = np.zeros((4, 2))
    t = 0
    while t == 0:  # check for correct order of peaks, otherwise increase window
        i = 0
        for k in [-1, 1]:
            # 1. find all peaks
            peaks_all, properties = scipy.signal.find_peaks(k * y, prominence=0.1, distance=0.05 * Fs,
                                                            width=0.01 * Fs)  #
            # 2. only get peaks with highest prominences
            thr = 0.9 * (properties["prominences"][np.argsort(properties["prominences"])[-np.min([len(peaks_all), 5])]])
            peaks, properties = scipy.signal.find_peaks(k * y, prominence=thr, distance=0.05 * Fs, width=0.010 * Fs)  #
            # remove peak if is <12ms after stim(probably artifact)
            # print((peaks+x1-t_0*Fs)/Fs)
            req = (peaks > 0.012 * Fs)
            peaks = peaks[req]
            for item in properties.items():
                properties[item[0]] = properties[item[0]][req]

            # if more than two peaks are found, take the two with the highest prominence
            if len(peaks) > 2:
                ix = (np.argsort(properties["prominences"])[-2:])
                peaks = peaks[ix]
                for item in properties.items():
                    properties[item[0]] = properties[item[0]][ix]
            peaks = peaks + x1
            properties["left_ips"] = properties["left_ips"] + x1
            properties["right_ips"] = properties["right_ips"] + x1
            # peaks in in s after stim
            peaks_s = (peaks - t_0 * Fs) / Fs
            # store peaks
            pk[i * 2:(i + 1) * 2, 0] = np.sort(peaks_s[0:2])
            pk[i * 2:(i + 1) * 2, 1] = x[np.sort(peaks[0:2])]
            i = i + 1
        if (all(np.argsort(pk[:, 0]) == [0, 2, 1, 3])) | (all(np.argsort(pk[:, 0]) == [2, 0, 3, 1])):
            t = 1
        else:
            x1 = int((t_0) * Fs)
            x2 = int(x2 + 0.3 * Fs)
            y = resp_all[x1:x2]
            pk = np.zeros((4, 2))
            i = 0
            if x2 > int((t_0 + 2) * Fs):
                pk = np.zeros((4, 2))
                t = 1
    pk = pk[np.argsort(pk[:, 0])]
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