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
import sys
import LL_funcs
import freq_funcs
from scipy.signal import butter,filtfilt

class main:
    def __init__(self, data, Fs, IPI, Int):
        self.IPI    = IPI
        self.Int    = Int
        self.Fs     = Fs
        self.EEG    = data

        ##filter
        nyq             = 0.5 * Fs  # Nyquist Frequency
        normal_cutoff   = 80 / nyq
        # Get the filter coefficients
        b, a        = butter(2, normal_cutoff, btype='low', analog=False)

        EEG_smooth                  = filtfilt(b, a, data) #butter_lowpass_filter(data=data, cutoff=80, fs=Fs, order=2)
        self.EEG_HP             = data-EEG_smooth
        self.pad                = np.sign(np.diff(np.pad(data, ((0, 0), (0, 0), (0, 1)), 'reflect')))
        self.pwr,  self.pwr_mean,  self.f, self.pha, self.pha_mean     = freq_funcs.get_WT(data, Fs=Fs)
        self.pwrT, self.pwr_meanT, self.f, self.phaT, self.pha_meanT = freq_funcs.get_WT(self.pad, Fs=Fs)
        self.pwrH, self.pwr_meanH, self.f, self.phaH, self.pha_meanH = freq_funcs.get_WT(self.EEG_HP, Fs=Fs)

        #self.fft_f, _, self.fft_pwr   = LL_funcs.get_WT(data, Fs=Fs)
        #self.fft_fT, _, self.fft_pwrT = LL_funcs.get_WT(self.pad, Fs=Fs)

    def plot_WT_resp(self, c, x_ax):
        fs = 14
        ls = 10

        self.resp_mean  = np.nanmean(self.EEG[0, :, :], axis=0)
        std = np.nanstd(self.EEG[0, :, :], axis=0)
        self.resp_meanT = np.nanmean(self.pad[0, :, :], axis=0)
        stdT = np.nanstd(self.pad[0, :, :], axis=0)
        self.resp_meanH = np.nanmean(self.EEG_HP[0, :, :], axis=0)
        stdH = np.nanstd(self.EEG_HP[0, :, :], axis=0)
        c = 0
        plt.subplot(3,3, 1) ### 3
        plt.plot(x_ax, self.resp_mean)
        plt.fill_between(x_ax, self.resp_mean - std,
                         self.resp_mean + std, alpha=0.1)
        plt.ylabel('[uV]', fontsize=fs)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.title('Raw EEG', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        #plt.xlim(-5, 5)
        plt.xlim(0, 3)

        plt.subplot(3,3,2)
        plt.plot(x_ax, self.resp_meanT*20, c=[1, 0, 0], alpha=0.5)
        plt.plot(x_ax, self.resp_mean)
        plt.ylabel('[uV]', fontsize=fs)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.title('Raw EEG, sign(diff)', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        # plt.xlim(-5, 5)
        plt.xlim(0, 3)

        plt.subplot(3, 3, 3)
        plt.plot(x_ax, self.resp_meanH)
        plt.ylabel('[uV]', fontsize=fs)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.title('Raw EEG, EEG - LP80Hz', fontsize=fs)
        plt.tick_params(axis='both', labelsize=ls)
        # plt.xlim(-5, 5)
        plt.xlim(0, 3)


        ax = plt.subplot(3,3, 4)
        plt.pcolormesh(x_ax, self.f, np.log(self.pwr_mean[:,c,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.title('WT transform', fontsize=fs)
        plt.tick_params(axis='both',labelsize=fs)
        # plt.xlim(-5, 5)
        plt.xlim(0, 3)

        ax = plt.subplot(3,3, 5)
        plt.pcolormesh(x_ax, self.f, np.log(self.pwr_meanT[:,c,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.title('WT transform, sign(diff)', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        # plt.xlim(-5, 5)
        plt.xlim(0, 3)

        ax = plt.subplot(3, 3, 6)
        plt.pcolormesh(x_ax, self.f, np.log(self.pwr_meanH[:, c, :]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.title('WT transform, EEG - LP80Hz', fontsize=fs)
        plt.tick_params(axis='both', labelsize=ls)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        # plt.xlim(-5, 5)
        plt.xlim(0, 3)

        ax = plt.subplot(3,3, 7)
        ax.pcolormesh(x_ax, self.f, np.log(self.pha_mean[:,c,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        plt.tick_params(axis='both',labelsize=ls)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        # plt.xlim(-5, 5)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.xlim(0, 3)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.xlabel('time [s]', fontsize=fs)
        plt.title('Phase', fontsize=fs)

        ax = plt.subplot(3,3, 8)
        plt.pcolormesh(x_ax, self.f, np.log(self.pha_meanT[:,c,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.xlabel('time [s]', fontsize=fs)
        plt.title('Phase, sign(diff)', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        # plt.xlim(-5, 5)
        plt.xlim(0, 3)

        ax = plt.subplot(3, 3, 9)
        plt.pcolormesh(x_ax, self.f, np.log(self.pha_meanH[:, c, :]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.xlabel('time [s]', fontsize=fs)
        plt.title('Phase, EEG - LP80Hz', fontsize=fs)
        plt.tick_params(axis='both', labelsize=ls)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        # plt.xlim(-5, 5)
        plt.xlim(0, 3)

        # ax = plt.subplot(4, 2, 7)
        # plt.plot(self.fft_f, self.fft_pwr[c, :])
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.xticks([0.5, 1, 50, 100, 150, 200], fontsize=15)
        # plt.xlabel('Frequency [Hz]', fontsize=20)
        # plt.ylabel('Power', fontsize=20)
        #
        # ax = plt.subplot(4, 2, 8)
        # plt.plot(self.fft_fT, self.fft_pwrT[c, :])
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.xticks([0.5, 1, 50, 100, 150, 200], fontsize=15)
        # plt.xlabel('Frequency [Hz]', fontsize=20)
        # plt.ylabel('Power', fontsize=20)

    def plot_WT_mean(self, c, x_ax):
        fs = 14
        ls = 10
        s = -3
        e = 3
        self.resp_mean  = np.nanmean(self.EEG[0, :, :], axis=0)
        self.resp_meanT = np.nanmean(self.pad[0, :, :], axis=0)
        self.resp_meanH = np.nanmean(self.EEG_HP[0, :, :], axis=0)
        c = 0
        plt.subplot(3,3, 1) ### 3
        plt.plot(x_ax, self.resp_mean)
        plt.ylabel('[uV]', fontsize=fs)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.title('Raw EEG', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        #plt.xlim(-5, 5)
        plt.xlim(s,e)

        plt.subplot(3,3,2)
        plt.plot(x_ax, self.resp_meanT*20, c=[1, 0, 0], alpha=0.5)
        plt.plot(x_ax, self.resp_mean)
        plt.ylabel('[uV]', fontsize=fs)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.title('Raw EEG, sign(diff)', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        # plt.xlim(-5, 5)
        plt.xlim(s,e)

        plt.subplot(3, 3, 3)
        plt.plot(x_ax, self.resp_meanH)
        plt.ylabel('[uV]', fontsize=fs)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.title('Raw EEG, EEG - LP80Hz', fontsize=fs)
        plt.tick_params(axis='both', labelsize=ls)
        # plt.xlim(-5, 5)
        plt.xlim(s,e)


        ax = plt.subplot(3,3, 4)
        plt.pcolormesh(x_ax, self.f, np.log(self.pwr_mean[:,c,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.title('WT transform', fontsize=fs)
        plt.tick_params(axis='both',labelsize=fs)
        # plt.xlim(-5, 5)
        plt.xlim(s,e)

        ax = plt.subplot(3,3, 5)
        plt.pcolormesh(x_ax, self.f, np.log(self.pwr_meanT[:,c,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.title('WT transform, sign(diff)', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        # plt.xlim(-5, 5)
        plt.xlim(s,e)

        ax = plt.subplot(3, 3, 6)
        plt.pcolormesh(x_ax, self.f, np.log(self.pwr_meanH[:, c, :]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.title('WT transform, EEG - LP80Hz', fontsize=fs)
        plt.tick_params(axis='both', labelsize=ls)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        # plt.xlim(-5, 5)
        plt.xlim(s,e)

        ax = plt.subplot(3,3, 7)
        ax.pcolormesh(x_ax, self.f, np.log(self.pha_mean[:,c,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        plt.tick_params(axis='both',labelsize=ls)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        # plt.xlim(-5, 5)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.xlim(s,e)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.xlabel('time [s]', fontsize=fs)
        plt.title('Phase', fontsize=fs)

        ax = plt.subplot(3,3, 8)
        plt.pcolormesh(x_ax, self.f, np.log(self.pha_meanT[:,c,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.xlabel('time [s]', fontsize=fs)
        plt.title('Phase, sign(diff)', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        # plt.xlim(-5, 5)
        plt.xlim(s,e)

        ax = plt.subplot(3, 3, 9)
        plt.pcolormesh(x_ax, self.f, np.log(self.pha_meanH[:, c, :]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.xlabel('time [s]', fontsize=fs)
        plt.title('Phase, EEG - LP80Hz', fontsize=fs)
        plt.tick_params(axis='both', labelsize=ls)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
        # plt.xlim(-5, 5)
        plt.xlim(s,e)

        # ax = plt.subplot(4, 2, 7)
        # plt.plot(self.fft_f, self.fft_pwr[c, :])
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.xticks([0.5, 1, 50, 100, 150, 200], fontsize=15)
        # plt.xlabel('Frequency [Hz]', fontsize=20)
        # plt.ylabel('Power', fontsize=20)
        #
        # ax = plt.subplot(4, 2, 8)
        # plt.plot(self.fft_fT, self.fft_pwrT[c, :])
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.xticks([0.5, 1, 50, 100, 150, 200], fontsize=15)
        # plt.xlabel('Frequency [Hz]', fontsize=20)
        # plt.ylabel('Power', fontsize=20)

    def plot_WT_trial(self, c, x_ax, stim):
        fs = 14
        ls = 12
        plt.subplot(3, 2, 1)
        plt.plot(x_ax, self.EEG[c,stim,:])
        plt.ylabel('[uV]', fontsize=20)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.title('Raw EEG', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        plt.xlim(-5, 5)
        #plt.xlim(-1, 3)

        plt.subplot(3, 2,2)
        plt.plot(x_ax, self.EEG[c, stim, :])
        plt.plot(x_ax, self.pad[c, stim, :] * 20, c=[1, 0, 0], alpha=0.5)
        plt.ylabel('[uV]', fontsize=20)
        plt.axvline((0 + self.IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
        plt.title('Raw EEG, sign(diff)', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        plt.xlim(-5, 5)
        # plt.xlim(-1, 3)

        ax = plt.subplot(3, 2, 3)
        plt.pcolormesh(x_ax, self.f, np.log(self.pwr[:,c,stim,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.title('WT transform', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        plt.xlim(-5, 5)
        # plt.xlim(-1, 3)

        ax = plt.subplot(3, 2, 4)
        plt.pcolormesh(x_ax, self.f, np.log(self.pwrT[:,c,stim,:]), cmap='jet')
        plt.yscale('log')
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.title('WT transform, sign(diff)', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)
        plt.xlim(-5, 5)
        # plt.xlim(-1, 3)

        ax = plt.subplot(3, 2, 5)
        ax.pcolormesh(x_ax, self.f, np.log(self.pha[:,c,stim,:]), cmap='jet')
        plt.yscale('log')
        plt.xlim(-5, 5)
        #plt.xlim(-1, 3)
        plt.yticks([1, 5, 20, 50, 200])
        plt.tick_params(axis='both',labelsize=ls)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.xlabel('time [s]', fontsize=fs)
        plt.title('Phase', fontsize=fs)

        ax = plt.subplot(3, 2, 6)
        plt.pcolormesh(x_ax, self.f, np.log(self.phaT[:,c,stim,:]), cmap='jet')
        plt.yscale('log')
        plt.xlim(-5, 5)
        # plt.xlim(-1, 3)
        plt.yticks([1, 5, 20, 50, 200])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.ylim(bottom=0.5, top=200)
        plt.ylabel('Frequency [Hz]', fontsize=fs)
        plt.xlabel('time [s]', fontsize=fs)
        plt.title('Phase, sign(diff)', fontsize=fs)
        plt.tick_params(axis='both',labelsize=ls)