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
import heartpy as hp
import seaborn as sns
import scipy.io as sio
from scipy.integrate import simps
import pandas as pd
from scipy import fft
import sys
import LL_funcs
class main:
    def __init__(self,subj, ChanP, StimChan, Fs, Int_all, IPI_all, resp, elec, recut=False):
        self.subj           = subj
        self.ChanP          = ChanP #39 for CL12
        self.StimLab        = StimChan #label
        self.Fs             = Fs
        self.Int_all        = Int_all
        self.IPI_all        = IPI_all
        #self.resp           = resp #which one of the response channel [0, 5, 8]
        self.resp = resp
        cwd = os.getcwd()
        self.path_patient  = os.path.dirname(os.path.dirname(cwd)) + '/Patients/' + subj

        self.stim_table     = pd.read_csv(self.path_patient + "/" + subj + "_stimulation_table_py.csv", sep=',') #;
        data                = pd.read_csv(self.path_patient +  "/" + subj + "_elec_table_BP.csv", header=None, dtype=str)
        self.labels_all     = data.values
        #todo: as input. subj specific
        self.elec           = elec
        self.dur = np.zeros((1, 2), dtype=np.int)
        self.dur[0, :] = [-5, 5]
        self.dur_tot = np.int(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        if recut:
            self.cut_data_all()
        else:
            self.load_data()

        # self.get_LL_baseline()
        #self.get_LL_baseline()

    def load_data(self):
        print('loading data....')
        self.stim_list = self.stim_table[(self.stim_table.h_block<25) &(self.stim_table.ChanP == self.ChanP) & (self.stim_table.noise == 0)]
        self.stim_list.insert(0, "Number", np.arange(len(self.stim_list)), True)
        try:
            # self.EEG_block = np.load('./Patients/' + self.subj + '/data_blocks/response/Resp_' + self.StimLab + '.npy')
            self.EEG_block = np.load(self.path_patient + '/data_blocks/response/All_' + self.StimLab + '.npy')

        except IOError:
            print('Data not found. Creating new file.... ')
            self.cut_data_all()

    def cut_data(self):
        self.stim_list = self.stim_table[(self.stim_table.h_block<25) &(self.stim_table.ChanP == self.ChanP) & (self.stim_table.noise == 0)]
        self.stim_list.insert(0, "Number", np.arange(len(self.stim_list)), True)
        self.EEG_block = np.zeros((len(self.resp), len(self.stim_list), self.dur_tot * self.Fs))
        self.EEG_block[:, :, :] = np.NaN
        ds = 0
        #todo: remove hrdcode
        ############ Noise
        k=-1
        for i in range(len(self.resp)):
            if self.labels_all[self.resp[i]] == 'FPIR01_FPIR02':
                k = i
                break
        for i in range(24):
            h = i + 1
            matfile = \
            h5py.File("./Patients/" + self.subj + "/data_blocks/time/" + self.subj + "_BP_" + str(h) + "_h_pp.mat", 'r')['EEGpp']
            EEGpp = matfile[()].T
            stim_list_h = self.stim_table[
                (self.stim_table.h_block == h) & (self.stim_table.ChanP == self.ChanP) & (self.stim_table.noise == 0)]

            for s in range(len(stim_list_h)):
                trig     = stim_list_h.TTL_DS.values[s]
                data_len = EEGpp[self.resp, trig + self.dur[0, 0] * self.Fs:trig + self.dur[0, 1] * self.Fs].shape[1]
                if data_len < self.dur_tot * self.Fs:
                    self.EEG_block[:, s + ds, 0:data_len] = EEGpp[self.resp, trig + self.dur[0, 0] * self.Fs:trig + self.dur[0, 1] * self.Fs]
                else:
                    self.EEG_block[:, s + ds, :] = EEGpp[self.resp, trig + self.dur[0, 0] * self.Fs:trig + self.dur[0, 1] * self.Fs]
                #todo: if FPIR1-2 -> put to NaN from hour 14
                if h>14 and k>=0:
                    self.EEG_block[k, s + ds, :] = np.NaN
            ds += s + 1
        np.save('./Patients/' + self.subj + '/data_blocks/response/Resp_' + self.StimLab + '.npy', self.EEG_block)
        print('Data block saved')

    def cut_data_all(self):
        self.stim_list = self.stim_table[
            (self.stim_table.h_block < 25) & (self.stim_table.ChanP == self.ChanP) & (self.stim_table.noise == 0)]
        self.stim_list.insert(0, "Number", np.arange(len(self.stim_list)), True)
        self.EEG_block = np.zeros((len(self.labels_all), len(self.stim_list), self.dur_tot * self.Fs))
        self.EEG_block[:, :, :] = np.NaN
        ds = 0
        # todo: remove hrdcode
        ############ Noise
        k = -1
        for i in range(len(self.labels_all)):
            if self.labels_all[i] == 'FPIR01_FPIR02':
                k = i
                break
        for i in range(22):
            h = i + 1
            matfile = \
                h5py.File(self.path_patient + "/data_blocks/time/" + self.subj + "_BP_" + str(h) + "_h_pp.mat",
                          'r')['EEGpp']
            EEGpp = matfile[()].T
            stim_list_h = self.stim_table[
                (self.stim_table.h_block == h) & (self.stim_table.ChanP == self.ChanP) & (self.stim_table.noise == 0)]

            for s in range(len(stim_list_h)):
                trig        = stim_list_h.TTL_DS.values[s]
                data_len    = EEGpp[self.resp, trig + self.dur[0, 0] * self.Fs:trig + self.dur[0, 1] * self.Fs].shape[1]
                if data_len < self.dur_tot * self.Fs:
                    self.EEG_block[:, s + ds, 0:data_len] = EEGpp[:,
                                                            trig + self.dur[0, 0] * self.Fs:trig + self.dur[
                                                                0, 1] * self.Fs]
                else:
                    self.EEG_block[:, s + ds, :] = EEGpp[:,
                                                   trig + self.dur[0, 0] * self.Fs:trig + self.dur[0, 1] * self.Fs]
                # todo: if FPIR1-2 -> put to NaN from hour 14
                if h > 14 and k >= 0:
                    self.EEG_block[k, s + ds, :] = np.NaN
                    #k = -1
            ds += s + 1
        np.save(self.path_patient + '/data_blocks/response/All_' + self.StimLab + '.npy', self.EEG_block)
        print('Data block saved')

    def get_LL_P2P(self):
        IPI = 0
        data_block = np.zeros((1, 8))
        for i in range(3):
            Int = 1 + i
            self.stim_list      = self.stim_table[(self.stim_table.h_block < 25) & (self.stim_table.ChanP == self.ChanP) & (self.stim_table.noise == 0)]
            a                   = self.stim_list [(self.stim_list .stim_block > 0) & (self.stim_list .Int_mA == Int) & (self.stim_list .Currentflow == 100)]
            stim_list_spef      = a[(a.IPI_ms == 0) | (a.IPI_ms > 315)]
            stimNum_SP          = stim_list_spef.Number.values
            stim_block          = stim_list_spef.stim_block.values
            data_SP             = self.EEG_block[:, stimNum_SP, :]
            # LL
            LL_resp, LL_max     = LL_funcs.get_LL_all(data=data_SP, Fs=self.Fs, win=0.1, t0= -self.dur[0, 0], IPI=0)
            pks, pks_loc        = LL_funcs.get_P2P_resp(data=data_SP, Fs=self.Fs, IPI=IPI, t_0=-self.dur[0, 0])
            for j in range(52):
                data_block_c = np.zeros((LL_max.shape[1], 8))
                data_block_c[:, 0] = j
                data_block_c[:, 1] = Int
                data_block_c[:, 2:4] = LL_max[j, :, :]
                data_block_c[:, 5] = pks[j, :, 2]
                data_block_c[:, 6] = stim_block
                data_block_c[:, 7] = stim_list_spef.StimNumber.values
                data_block = np.concatenate([data_block, data_block_c], axis=0)

        data_block = data_block[1:-1, :]
        LL_all = pd.DataFrame(
            {"Chan": data_block[:, 0], "Int [mA]": data_block[:, 1], "LL": data_block[:, 2], "LL_loc": data_block[:, 3],
             "P2P": data_block[:, 5], "Stim Block": data_block[:, 6], "Stim Num": data_block[:, 7]})

        LL_all.to_csv(self.path_patient + '/Analysis/LL/LL_P2P_SP_' + StimChan + '.csv', index=False,
                      header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    def total_resp_avg(self, c):
        stim_list_spef = self.stim_list
        stimNum1       = stim_list_spef.Number.values
        resps          = np.zeros((len(stimNum1), 4 * Fs))
        k = 0
        for i in range(len(self.IPI_all)):
            IPI = self.IPI_all[i]
            t_0 = np.int((5 + IPI / 1000 - 0.1) * self.Fs)
            t_1 = np.int(t_0 + 4 * self.Fs)
            stim_list_spef = self.stim_list[(self.stim_list.IPI_ms == IPI)]  # &(stim_list.Int_mA == 3)
            stimNum = stim_list_spef.Number.values
            j       = len(stimNum)
            resps[k:k + j, :] = self.EEG_block[c, stimNum, t_0:t_1]
            k = k + j
        resp_mean   = np.nanmean(resps, axis=0)
        resp_std    = np.nanstd(resps, axis=0)
        x_ax_resp   = np.arange(-0.1, 3.9, (1 / Fs))
        max_am      = np.nanquantile(abs(resps), 0.995)
        plt.figure(figsize=(10, 5))
        plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.title(
            str('Mean Resp: ' + self.StimLab + ' Stim, ' + self.labels_all[self.resp[c]] + ', Num: ' + str(len(stimNum1)))[2:-2],
            fontsize=15)
        plt.plot(x_ax_resp, resp_mean)
        plt.fill_between(x_ax_resp, resp_mean - resp_std, resp_mean + resp_std, alpha=0.3)
        plt.xlim([-0.1, 2])
        plt.xlabel('time [s]')
        plt.ylim([-max_am, max_am])
        plt.ylabel('uV')

    def int_on_SP(self, c):
        #get mean and std
        IPI             = 0
        resp_mean       = np.zeros((3, self.dur_tot * self.Fs))
        resp_std        = np.zeros((3, self.dur_tot * self.Fs))
        num =  np.zeros((3,))
        for i in range(3):
            Int             = i + 1
            a  = self.stim_list[(self.stim_list.Int_mA == Int)] #(self.stim_list.IPI_ms == IPI) &
            stim_list_spef = a[(a.IPI_ms==0) |(a.IPI_ms>300)]
            stimNum         = stim_list_spef.Number.values
            resp_mean[i, :] = np.nanmean(self.EEG_block[c, stimNum, :], axis=0)
            resp_std[i, :]  = np.nanstd(self.EEG_block[c, stimNum, :], axis=0)
            num[i] = len(stimNum)

        #plot
        plt.figure(figsize=(20, 15))
        plt.subplot(2, 1, 1)
        plt.title(str('SP - Mean: ' + self.StimLab + ' Stim, ' + self.labels_all[self.resp[c]])[2:-2], fontsize=20)
        plt.plot(self.x_ax, resp_mean[0], label='1mA, n='+str(num[0]))
        plt.fill_between(self.x_ax, resp_mean[0] - resp_std[0], resp_mean[0] + resp_std[0], alpha=0.3)
        plt.plot(self.x_ax, resp_mean[1], label='2mA, n='+str(num[1]))
        plt.fill_between(self.x_ax, resp_mean[1] - resp_std[1], resp_mean[1] + resp_std[1], alpha=0.3)
        plt.plot(self.x_ax, resp_mean[2], label='3mA, n='+str(num[2]))
        plt.fill_between(self.x_ax, resp_mean[2] - resp_std[2], resp_mean[2] + resp_std[2], alpha=0.3)
        plt.ylabel('[uV]', fontsize=20)
        plt.axvline(IPI / 1000, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.axvspan(-0.1, 2, facecolor='b', alpha=0.015)
        plt.tick_params(axis="x", labelsize=20)
        plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.xlim([-3, 5])
        plt.ylim([-800, 800])
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.x_ax, resp_mean[0], label='1mA')
        plt.fill_between(self.x_ax, resp_mean[0] - resp_std[0], resp_mean[0] + resp_std[0], alpha=0.3)
        plt.plot(self.x_ax, resp_mean[1], label='2mA')
        plt.fill_between(self.x_ax, resp_mean[1] - resp_std[1], resp_mean[1] + resp_std[1], alpha=0.3)
        plt.plot(self.x_ax, resp_mean[2], label='3mA')
        plt.fill_between(self.x_ax, resp_mean[2] - resp_std[2], resp_mean[2] + resp_std[2], alpha=0.3)
        plt.xlabel('time [s]', fontsize=20)
        plt.ylabel('[uV]', fontsize=20)
        plt.tick_params(axis="x", labelsize=20)
        plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.axvline(IPI / 1000, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.legend()
        plt.xlim([-0.1, 2])
        plt.show()
    def IPI_resp(self, c, IPI_resp, Int):
        #get mean and std
        resp_mean       = np.zeros((len(IPI_resp), self.dur_tot * self.Fs))
        resp_std        = np.zeros((len(IPI_resp), self.dur_tot * self.Fs))
        x_ax_resp       = np.arange(-3, 5, (1 / Fs))
        # plot
        plt.figure(figsize=(20, 5))
        plt.title(
            str('SP - Mean: ' + self.StimLab + ' Stim, ' + self.labels_all[self.resp[c]])[2:-2],
            fontsize=20)
        for i in range(len(IPI_resp)):
            IPI             = IPI_resp[i]
            t_0             = np.int((5 + IPI / 1000 - 3) * self.Fs)
            t_1             = np.int(t_0 + 8 * self.Fs)
            stim_list_spef  = self.stim_list[(self.stim_list.IPI_ms == IPI) & (self.stim_list.Int_mA == Int)]
            stimNum         = stim_list_spef.Number.values
            resp_mean[i, :] = np.nanmean(self.EEG_block[c, stimNum, t_0:t_1], axis=0)
            resp_std[i, :]  = np.nanstd(self.EEG_block[c, stimNum, t_0:t_1], axis=0)
            plt.subplot(2, 1, 1)
            plt.plot(x_ax_resp, resp_mean[i, :], label=str(IPI) + 'ms')
            plt.fill_between(x_ax_resp, resp_mean[i, :] - resp_std[i, :], resp_mean[i, :] + resp_std[i, :], alpha=0.1)
            plt.ylabel('[uV]', fontsize=20)
            plt.axvspan(-0.1, 2, facecolor='b', alpha=0.015)
            plt.tick_params(axis="x", labelsize=20)
            plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
            plt.xlim([-3, 5])
            plt.ylim([-800, 800])
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(x_ax_resp, resp_mean[i, :], label=str(IPI) + 'ms')
            plt.fill_between(x_ax_resp, resp_mean[i, :] - resp_std[i, :], resp_mean[i, :] + resp_std[i, :], alpha=0.1)

            plt.xlabel('time [s]', fontsize=20)
            plt.ylabel('[uV]', fontsize=20)
            plt.tick_params(axis="x", labelsize=20)
            plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)

            plt.legend()
            plt.xlim([-0.1, 2])
    def int_on_PP(self, c, IPI):
        #get mean and std
        resp_mean       = np.zeros((3, self.dur_tot * self.Fs))
        resp_std        = np.zeros((3, self.dur_tot * self.Fs))
        for i in range(3):
            Int             = i + 1
            stim_list_spef  = self.stim_list[(self.stim_list.IPI_ms == IPI) & (self.stim_list.Int_mA == Int)]
            stimNum         = stim_list_spef.Number.values
            resp_mean[i, :] = np.nanmean(self.EEG_block[c, stimNum, :], axis=0)
            resp_std[i, :]  = np.nanstd(self.EEG_block[c, stimNum, :], axis=0)

        #plot
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(str("PP - Mean: " + self.StimLab + "Stim, " + self.labels_all[self.resp[c]] + ", IPI: " + str(IPI) + "ms, Num:" + str(len(stimNum)))[2:-2], fontsize=20)

        plt.subplot(2, 1, 1)

        plt.plot(self.x_ax, resp_mean[0], label='1mA')
        plt.fill_between(self.x_ax, resp_mean[0] - resp_std[0], resp_mean[0] + resp_std[0], alpha=0.3)
        plt.plot(self.x_ax, resp_mean[1], label='2mA')
        plt.fill_between(self.x_ax, resp_mean[1] - resp_std[1], resp_mean[1] + resp_std[1], alpha=0.3)
        plt.plot(self.x_ax, resp_mean[2], label='3mA')
        plt.fill_between(self.x_ax, resp_mean[2] - resp_std[2], resp_mean[2] + resp_std[2], alpha=0.3)
        plt.ylabel('[uV]', fontsize=20)
        plt.axvline(IPI / 1000, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.axvspan(-0.1, 2, facecolor='b', alpha=0.015)
        plt.tick_params(axis="x", labelsize=20)
        plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.xlim([-3, 5])
        plt.ylim([-800, 800])
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.x_ax, resp_mean[0], label='1mA')
        plt.fill_between(self.x_ax, resp_mean[0] - resp_std[0], resp_mean[0] + resp_std[0], alpha=0.3)
        plt.plot(self.x_ax, resp_mean[1], label='2mA')
        plt.fill_between(self.x_ax, resp_mean[1] - resp_std[1], resp_mean[1] + resp_std[1], alpha=0.3)
        plt.plot(self.x_ax, resp_mean[2], label='3mA')
        plt.fill_between(self.x_ax, resp_mean[2] - resp_std[2], resp_mean[2] + resp_std[2], alpha=0.3)
        plt.xlabel('time [s]', fontsize=20)
        plt.ylabel('[uV]', fontsize=20)
        plt.tick_params(axis="x", labelsize=20)
        plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.axvline(IPI / 1000, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.legend()
        plt.xlim([-0.1, 2])

    def mean_resp(self, c, IPI, Int):
        #get specific stimulation from list
        stim_list_spef      = self.stim_list[(self.stim_list.IPI_ms == IPI) & (self.stim_list.Int_mA == Int)]
        stimNum             = stim_list_spef.Number.values

        resp_mean           = np.nanmean(self.EEG_block[c, stimNum, :], axis=0)
        plt.figure(figsize=(30, 5))
        plt.plot(self.x_ax, resp_mean, c=[0.104, 0.306, 0.472], linewidth=1.5)
        plt.xlabel('time [s]', fontsize=20)
        plt.ylabel('[uV]', fontsize=20)
        plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.axvline(IPI / 1000, ls='--', c=[1, 0, 0], linewidth=1.2)
        plt.ylim([-300, 300])
        plt.tick_params(labelsize=20)
        plt.title('Mean: ' + self.StimLab + ' Stim, ' + self.labels_all[self.resp[c]] + ' Resp, Int: ' + str(
            Int) + 'mA, IPI:' + str(IPI) + ' ms, Num: ' + str(len(stimNum)), fontsize=20)

    def plot_prop_mean(self, IPI, Int):
        stim_list_spef      = self.stim_list[(self.stim_list.IPI_ms == IPI) & (self.stim_list.Int_mA == Int)]
        stimNum             = stim_list_spef.Number.values
        resp_mean           = np.nanmean(self.EEG_block[:, stimNum, :], axis=1)
        shape_resps         = resp_mean.shape
        fig, axes = plt.subplots(3, 2, figsize=(20, 14))
        #fig                 = plt.figure()
        if IPI == 0:
             fig.suptitle(self.subj + ": "+ self.StimLab + " SP - Stimulation, Int: " + str(Int) + "mA, Stims:"+str(len(stimNum)), fontsize=25)
        else:
             fig.suptitle(self.subj + ": "+ self.StimLab +" PP - Stimulation, IPI: " + str(IPI) + "ms, Int: " + str(Int) + "mA, Stims:"+str(len(stimNum)), fontsize=25)

        for r in range(len(self.resp)):
            num_let = (len(self.labels_all[self.resp[r], 0]) - 5) / 2
            if self.labels_all[r,0] !=self.StimLab and self.labels_all[r,0][0:np.int(num_let+2)] not in self.StimLab and  self.labels_all[r,0][-np.int(num_let+2):] not in self.StimLab:
                #get which electrode the response channel is from

                for item in self.elec:
                    if str(self.labels_all[self.resp[r],0])[0:np.int(num_let)] in item: #str(self.labels_all[self.resp[r]])[-6:-4]
                        fig_num    = self.elec.index(item)
                        elec_label = item
                        #print(elec_label, fig_num)
                        break

                plt.subplot(len(self.elec)/2, 2, fig_num + 1)
                plt.plot(self.x_ax, resp_mean[r, :], label=str(self.labels_all[self.resp[r]])[2:-2], linewidth=1.5)
                plt.ylabel('[uv]', fontsize=15)
                plt.title(elec_label, fontsize=20)
                if fig_num == len(self.elec)-1 or fig_num == len(self.elec):
                    plt.xlabel('time [s]', fontsize=15)
                plt.xlim([-0.5, 2])
                plt.ylim([-600,600])
                plt.axvline(x=0, ls='--', c=[0, 0, 0], linewidth=0.5)
                plt.axvline(x=0 + IPI / 1000, ls='--', c=[0, 0, 0], linewidth=0.5)
                plt.legend(fontsize=12, loc='upper right')
        #fig.show()
        plt.show()
    def mean_all_PP(self, c):
        # get max value for y-axis limits
        Int            = 3
        stim_list_spef = self.stim_list[(self.stim_list.Int_mA == Int)&(self.stim_list.IPI_ms == 1000)]
        stimNum        = stim_list_spef.Number.values
        # numberofStims = stim_list_spef.shape[0]
        #limy           = np.nanpercentile(self.EEG_block[c, stimNum, :], 90)
        limy           = np.max(np.nanmean(self.EEG_block[c, stimNum, :], axis=0))
        limy           = limy + 0.4 * limy

        resp_mean_IPI   = np.zeros((3, len(self.IPI_all), self.dur_tot * self.Fs))
        fig             = plt.figure(figsize=(20, 20))
        fig.suptitle('Mean -- ' + self.StimLab+ ' Stim, ' + self.labels_all[self.resp[c]] + ' Resp', fontsize=17)
        for k in range(3):
            Int         = self.Int_all[k]
            for i in range(len(self.IPI_all)):
                IPI                     = self.IPI_all[i]
                #specific stimulations form list
                stim_list_spef          = self.stim_list[(self.stim_list.IPI_ms == IPI) & (self.stim_list.Int_mA == Int)]
                stimNum                 = stim_list_spef.Number.values
                #numberofStims   = stim_list_spef.shape[0]
                resp_mean_IPI[k, i, :]  = np.nanmean(self.EEG_block[c, stimNum, :], axis=0)
                plt.subplot(len(self.IPI_all), 3, i * 3 + k + 1)
                plt.plot(self.x_ax, resp_mean_IPI[k, i, :])
                plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=0.7)
                plt.axvline(IPI / 1000, ls='--', c=[1, 0, 0], linewidth=0.7)
                plt.ylabel(IPI, fontsize=10)
                plt.xlim([-0.1, 2])
                #todo: remove hardcode - maximum of 3mA
                plt.ylim([-limy, limy])
                plt.yticks([])
                if i * 3 + k + 1 < 4:
                    plt.title(str(Int) + 'mA', fontsize=17)
                if i * 3 + k + 1 > 90:
                    plt.xlabel('time [s]', fontsize=15)
                    plt.tick_params(axis="x", labelsize=15)
                else:
                    plt.xticks([])

    def morlet_SP(self, Int):
        IPI = 0
        #specific stimulations form list
        stim_list_spef          = self.stim_list[(self.stim_list.IPI_ms == IPI) & (self.stim_list.Int_mA == Int)]
        self.stimNum_SP         = stim_list_spef.Number.values
        #todo:  how to transform data ????
        self.EEG_wtSP, self.EEG_wt_meanSP,self.freqsSP,_,_ = LL_funcs.get_Wt(data=self.EEG_block[:,self.stimNum_SP,:], Fs=self.Fs)

        self.pksSP, self.locSP                         = LL_funcs.get_P2P_resp(data=self.EEG_block[:,self.stimNum_SP,:], Fs=self.Fs, IPI=IPI, t_0=(-self.dur[0, 0]))
        self.LL_SP                                     = LL_funcs.get_LL_resp(data=self.EEG_block[:,self.stimNum_SP,:], wdp_S=0.5, Fs=self.Fs)

    def morlet_PP(self, Int, IPI):
        #specific stimulations form list
        stim_list_spef          = self.stim_list[(self.stim_list.IPI_ms == IPI) & (self.stim_list.Int_mA == Int)]
        self.stimNum_PP          = stim_list_spef.Number.values
        self.EEG_wtPP, self.EEG_wt_meanPP,self.freqsPP,_,_  = LL_funcs.get_WT(data=self.EEG_block[:,self.stimNum_PP,:], Fs=self.Fs)
        self.pks, self.loc                              = LL_funcs.get_P2P_resp(data=self.EEG_block[:,self.stimNum_PP,:], Fs=self.Fs, IPI=IPI, t_0=(-self.dur[0, 0]))
        self.LL_PP                            = LL_funcs.get_LL_resp(data=self.EEG_block[:, self.stimNum_PP, :], wdp_S=0.5, Fs=self.Fs)
        #self.pk_SP, self.loc_pk_SP                     = LL_funcs.get_P2P_resp(data=self.EEG_block[:, self.stimNum_SP, :], Fs=self.Fs, IPI=IPI)

    def plot_LL_PP_trial(self, c, Int, IPI):
        #self.morlet_PP(Int=Int, IPI=IPI)
        num             = 7
        fig, axarr      = plt.subplots(num, 4, figsize=(40, 15),
                                  gridspec_kw={'width_ratios': [1,2,2,1]})  # , figsize=(10,15), sharex=True
        fig.suptitle(self.subj + ', ' + self.StimLab + ' Stimulation,' + self.labels_all[self.resp[c]] + ', IPI: ' + str(IPI) + 'ms, Int: ' + str(Int) + 'mA', fontsize=20)
        fac = 1
        #short axis for SP
        durSP           = np.zeros((1, 2), dtype=np.int)
        durSP[0, :]     = [-1, 4]
        # dur_tot         = np.int(np.sum(abs(durSP))) # 2.5s
        x_axSP          = np.arange(durSP[0, 0], durSP[0, 1], (1 / self.Fs))

        if IPI == 0 or IPI == 30.4 or IPI == 103.2: fac = 10
        for i in range(num):
            if i == 0:
                axarr[i, 0].set_title('Raw EEG - SP', fontsize=20)
                axarr[i, 1].set_title('Raw EEG - PP ', fontsize=20)
                axarr[i, 2].set_title('spectro - PP  ', fontsize=20)
                axarr[i, 3].set_title('spectro - SP  ', fontsize=20)
            axarr[i, 1].plot(self.x_ax, self.EEG_block[c, self.stimNum_PP[i * fac], :], linewidth=2)
            #todo: x_ax_SP
            axarr[i, 0].plot(self.x_ax, self.EEG_block[c, self.stimNum_SP[i * 10], :], linewidth=2)#np.int(-(self.dur[0,0]-durSP[0, 0]) * self.Fs):np.int(-(self.dur[0,0]-durSP[0, 1]) * self.Fs)

            axarr[i, 0].plot(self.locSP[c, i * 10, 0] / self.Fs, self.pksSP[c, i * 10, 0], 'ro', markersize=4.5)
            axarr[i, 0].plot(self.locSP[c, i * 10, 1] / self.Fs, self.pksSP[c, i * 10, 1], 'ro', markersize=4.5)
            axarr[i, 1].plot(self.loc[c, i * fac, 0] / self.Fs, self.pks[c, i * fac, 0], 'ro', markersize=4.5)
            axarr[i, 1].plot(self.loc[c, i * fac, 1] / self.Fs, self.pks[c, i * fac, 1], 'ro', markersize=4.5)
            axarr[i, 1].axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
            axarr[i, 0].axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
            axarr[i, 1].axvline((0 + IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
            axarr[i, 0].set_ylabel('[uV]')
            axarr[i, 1].set_ylabel('[uV]')

            axarr[i, 2].plot(self.x_ax, self.LL_PP[c,i*fac,:], linewidth=2)
            axarr[i, 3].plot(self.x_ax, self.LL_SP[c, i * 10, :], linewidth=2)

            # axarr[i, 1].axvspan((IPI/1000)+0.02, (IPI/1000)+0.52, facecolor='b', alpha=0.2)
            axarr[i, 1].set_xlim([-5, 5])
            axarr[i, 2].set_xlim([-5, 5])
            axarr[i, 0].set_xlim([-1, 4])
            axarr[i, 3].set_xlim([-1, 4])

            axarr[i, 0].set_ylim([-300, 300])
            axarr[i, 1].set_ylim([-300, 300])

            if i == num - 1:
                axarr[i, 0].set_xlabel('time [s]', fontsize=16)
                axarr[i, 1].set_xlabel('time [s]', fontsize=16)
                axarr[i, 2].set_xlabel('time [s]', fontsize=16)
                axarr[i, 3].set_xlabel('time [s]', fontsize=16)

    def plot_morlet_PP_trial(self, c, Int, IPI):
        num = 7
        fig, axarr = plt.subplots(num, 4, figsize=(40, 15),
                                  gridspec_kw={'width_ratios': [1, 2, 2, 1]})  # , figsize=(10,15), sharex=True
        fig.suptitle(
            self.subj + ', ' + self.StimLab + ' Stimulation,' + self.labels_all[self.resp[c]] + ', IPI: ' + str(
                IPI) + 'ms, Int: ' + str(Int) + 'mA', fontsize=20)
        fac = 1
        # short axis for SP
        durSP = np.zeros((1, 2), dtype=np.int)
        durSP[0, :] = [-1, 4]
        # dur_tot         = np.int(np.sum(abs(durSP))) # 2.5s
        x_axSP = np.arange(durSP[0, 0], durSP[0, 1], (1 / self.Fs))

        if IPI == 0 or IPI == 30.4 or IPI == 103.2: fac = 10
        for i in range(num):
            if i == 0:
                axarr[i, 0].set_title('Raw EEG - SP', fontsize=20)
                axarr[i, 1].set_title('Raw EEG - PP ', fontsize=20)
                axarr[i, 2].set_title('LL - PP  ', fontsize=20)
                axarr[i, 3].set_title('LL - SP  ', fontsize=20)
            axarr[i, 1].plot(self.x_ax, self.EEG_block[c, self.stimNum_PP[i * fac], :], linewidth=2)
            # todo: x_ax_SP
            axarr[i, 0].plot(self.x_ax, self.EEG_block[c, self.stimNum_SP[i * 10],:], linewidth=2)

            axarr[i, 0].plot(self.locSP[c, i * 10, 0] / self.Fs, self.pksSP[c, i * 10, 0], 'ro', markersize=4.5)
            axarr[i, 0].plot(self.locSP[c, i * 10, 1] / self.Fs, self.pksSP[c, i * 10, 1], 'ro', markersize=4.5)
            axarr[i, 1].plot(self.loc[c, i * fac, 0] / self.Fs, self.pks[c, i * fac, 0], 'ro', markersize=4.5)
            axarr[i, 1].plot(self.loc[c, i * fac, 1] / self.Fs, self.pks[c, i * fac, 1], 'ro', markersize=4.5)
            axarr[i, 1].axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
            axarr[i, 0].axvline(0, ls='--', c=[0.8, 0, 0], linewidth=1)
            axarr[i, 1].axvline((0 + IPI) / 1000, ls='--', c=[0.8, 0, 0], linewidth=1)
            axarr[i, 0].set_ylabel('[uV]')
            axarr[i, 1].set_ylabel('[uV]')

            axarr[i, 2].pcolormesh(self.x_ax, self.freqsPP, np.log(self.EEG_wtPP[:, c, i * fac, :]), cmap='jet')
            axarr[i, 2].set_yscale('log')
            axarr[i, 2].set_yticks([1, 5, 20, 50, 100, 200])
            axarr[i, 2].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axarr[i, 2].set_ylim(bottom=1, top=200)
            axarr[i, 2].set_ylabel('Frequency [Hz]')

            axarr[i, 3].pcolormesh(self.x_ax, self.freqsPP, np.log(self.EEG_wtSP[:, c, i * 10, :]),
                                   cmap='jet')
            axarr[i, 3].set_yscale('log')
            axarr[i, 3].set_yticks([1, 5, 20, 50, 200])
            axarr[i, 3].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axarr[i, 3].set_ylim(bottom=1, top=200)
            axarr[i, 3].set_ylabel('Frequency [Hz]')

            # axarr[i, 1].axvspan((IPI/1000)+0.02, (IPI/1000)+0.52, facecolor='b', alpha=0.2)
            axarr[i, 1].set_xlim([-5,5])
            axarr[i, 2].set_xlim([-5,5])
            axarr[i, 0].set_xlim([-1,4])
            axarr[i, 3].set_xlim([-1,4])

            axarr[i, 0].set_ylim([-300, 300])
            axarr[i, 1].set_ylim([-300, 300])

            if i == num - 1:
                axarr[i, 0].set_xlabel('time [s]', fontsize=16)
                axarr[i, 1].set_xlabel('time [s]', fontsize=16)
                axarr[i, 2].set_xlabel('time [s]', fontsize=16)
                axarr[i, 3].set_xlabel('time [s]', fontsize=16)
