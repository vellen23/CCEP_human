import os
import numpy as np
import mne
import h5py
#import scipy.fftpack
import matplotlib
import pywt
from matplotlib.ticker import ScalarFormatter
import platform
import matplotlib.pyplot as plt
#from scipy import signal
import time
import seaborn as sns
#import scipy.io as sio
#from scipy.integrate import simps
import pandas as pd
#from scipy import fft
import matplotlib.mlab as mlab
import sys
sys.path.append('./py_functions')
import analys_func
#import PCA_funcs
from matplotlib.gridspec import GridSpec
import mne
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy import signal
from sklearn.decomposition import PCA, FastICA

import LL_funcs
from pandas import read_excel
from scipy.stats import norm

class main:
    def __init__(self,subj):
        if platform.system()=='Windows':
            sep = ','
        else: #'Darwin' for MAC
            sep =';'

        self.color_elab = np.zeros((4, 3))
        self.color_elab[0, :] = np.array([35, 25, 34]) / 255  # np.array([31, 78, 121]) / 255
        self.color_elab[1, :] = np.array([95, 75, 60]) / 255
        self.color_elab[2, :] = np.array([56, 73, 59]) / 255  # np.array([0.256, 0.574, 0.431])
        self.color_elab[3, :] = np.array([0.81, 0.33, 0.23])

        self.subj           = subj
        cwd                 = os.getcwd()
        self.path_patient   = os.path.dirname(os.path.dirname(cwd)) + '/Patients/' + subj
        labels_all          = pd.read_csv(self.path_patient + "/infos/" + subj + "_BP_labels.csv", header=0, dtype=str, sep=sep)
        self.labels_all     = labels_all.label.values
        self.cat_all        = labels_all.Cat.values
        data                = pd.read_csv(self.path_patient + "/infos/" + subj + "_BP_labels.csv", header=0, sep=sep)
        file_name           = subj + '_lookup.xlsx'
        df                  = pd.read_excel(os.path.join(self.path_patient + "/infos/", file_name),sheet_name='Par_benzo')  # ChanP  ChanN  Int [mA]  IPI [ms]  ISI [s]  Num_P
        stim_chan           = np.array(df.values[:, 0:2], dtype=float)
        stim_chan           = stim_chan[~np.isnan(stim_chan)]
        stim_chan           = stim_chan[np.nonzero(stim_chan)].reshape(-1, 2)
        self.StimChanNums   = stim_chan[:, 0]

        self.StimChans = []  # np.zeros((len(stim_chan)))
        for i in range(len(stim_chan)):
            self.StimChans.append(self.labels_all[(np.array(data.chan_BP_P.values) == stim_chan[i, 0]) & (np.array(data.chan_BP_N.values) == stim_chan[i, 1])][0])

        self.stim_ind = np.where(np.in1d(self.labels_all, self.StimChans))[0]

        #self.IPI_all = df.IPI.values
        Int_all             = df.SP_Int.values
        self.Int_all        = np.sort(Int_all[~np.isnan(Int_all)])
        # Int_p = df.Int_prob.values
        # self.Int_p = Int_p[~np.isnan(Int_p)]

        self.Fs = 500 #todo: remove hardcode
        self.dur = np.zeros((1, 2), dtype=np.int)
        self.dur[0, :] = [-5, 5]
        self.dur_tot = np.int(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        self.colors_Int = np.zeros((len(self.Int_all), 3))
        self.colors_Int[:, 0] = np.linspace(0, 1, len(self.Int_all))

        self.t_all = ['IO1', 'IO2', 'IO3']  # ['IO1', 'IO2', 'IO3']
        self.t_label = ['Baseline', 'Flumazenil', 'Benzodiazepin']  # ['Baseline', 'Flumazenil', 'Benzodiazepin']


    def cut_blocks_stim(self, exp, protocols):
        for i in protocols: #which protocols, e.g. protocols=[0,2], only IO1 and IO3 resp BL and Benzo
            t           = self.t_all[i]
            stim_table = pd.read_excel(self.path_patient + '/Data/experiment' + str(exp) + '/' + self.subj + "_stimlist_Ph_IO.xlsx",
                sheet_name=str(i + 1))
            stim_table.insert(0, "Number", np.arange(len(stim_table)), True)
            EEG_block           = np.zeros((len(self.labels_all), len(stim_table), self.dur_tot * self.Fs))
            EEG_block[:, :, :]  = np.NaN
            print(t)
            matfile = h5py.File(self.path_patient + '/Data/experiment' + str(exp) + '/data_blocks/time/' +self.subj + "_BP_Ph_" + t + "_pp/"+ self.subj + "_BP_Ph_" + t + "_pp.mat",
                      'r')['EEGpp']
            EEGpp = matfile[()].T
            for s in range(len(stim_table)):
                trig = stim_table.TTL_DS.values[s]
                EEG_block[:, s, :] = EEGpp[:, np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]

            stim_table.to_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv',index=False, header=True)
            np.save(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy', EEG_block)
            print('Data block saved \n')

    def get_LL_sc(self, sc, protocols, w):
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        print(StimChan)
        #w               = 0.1
        data_LL         = np.zeros((1, 5))  # RespChan, Int, LL, LLnorm, State
        mx = np.max(self.Int_all)
        for j in protocols:
            t               = self.t_all[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)]
            stimNum         = stim_list_spef.Number.values

            IPIs            = np.array(stim_list_spef.IPI_ms.values)
            LL              = LL_funcs.get_LL_both(data=EEG_block[:, stimNum, :], Fs=500, IPI=IPIs, t_0=5, win=w)

            LL_BL           = np.zeros((len(self.labels_all), 2))  # mean and std LL of each response channel
            LLb             = LL_funcs.get_LL_both(data=EEG_block[:, stimNum, :], Fs=500, IPI=IPIs, t_0=3, win=w)
            LL_BL[:, 0]     = np.nanmean(LLb[:, :, 1], axis=1)
            LL_BL[:, 1]     = np.nanstd(LLb[:, :, 1], axis=1)
            for c in range(len(LL)):
                val         = np.zeros((LL.shape[1], 5))
                val[:, 0]   = c                                         # response channel
                val[:, 1]   = stim_list_spef.Int_prob.values            # probing intensity
                val[:, 2]   = LL[c, :, 1]                               # absolut value of response
                val[:, 3]   = (LL[c, :, 1] - LL_BL[c, 0]) / LL_BL[c, 1] # Z - score
                val[:, 4]   = j                                         # protocol number


                data_LL = np.concatenate((data_LL, val), axis=0)

        data_LL = data_LL[1:-1, :]
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Int": data_LL[:, 1], "LL": data_LL[:, 2], "LL Zscore": data_LL[:, 3],
             "State": data_LL[:, 4]})
        LL_all.insert(3, "Condition", np.repeat(self.t_label[0], LL_all.shape[0]), True)
        LL_all.insert(3, "LL norm", 0, True)
        for c in range(len(self.labels_all)):
            ref = np.mean(LL_all[(LL_all.Int==mx)&(LL_all.State==0)&(LL_all.Chan==c)]['LL'])
            LL_all.loc[(LL_all.Chan == c), 'LL norm'] = LL_all.loc[(LL_all.Chan == c)]['LL']/ref
        for p in range(len(self.t_all)):
            LL_all.loc[(LL_all.State == p), 'Condition'] = self.t_label[p]
        LL_all.to_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan +'_'+str(w)+'s.csv', index=False,
                      header=True)  # scat_plot = scat_plot.fillna(method='ffill')
        print('Data saved')
    def IO_prot(self, sc, c, w):
        ## GABA paper

        #plot all mean, intensity by color gradient
        StimChan        = self.StimChans[sc]
        #ChanP           = self.StimChanNums[sc]
        LL_all          = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan +'_'+str(w)+'s.csv')
        fig             = plt.figure(figsize=(10, 8))
        plt.suptitle('LL '+str(w)+'s --  Stim: ' + StimChan + ', Resp: ' + self.labels_all[c], fontsize=16)
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        sns.pointplot(x="Int", y="LL", hue="Condition", data=LL_all[LL_all.Chan == c], kind="point", ci="sd",
                      height=6,  # make the plot 5 units high
                      aspect=1.5, s=0.8, legend_out=False,
                      palette=sns.color_palette([self.color_elab[0, :], self.color_elab[2, :]]))
        plt.xlabel('Int [mA]', fontsize=16)
        plt.ylabel('LL [uV/ms]', fontsize=16)
        plt.tick_params(axis="both", labelsize=16)
        plt.legend(fontsize=16)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IO/IO_prots_' + StimChan + '-' + str(
            self.labels_all[c]) + '_'+str(w)+'s.jpg')
        plt.show()
    def IO_prot_norm_cat(self, sc, cat, w, PP=False):
        # plot LL normalized by max for all channels within a specific brain area (cat, category)
        ## GABA paper

        #plot all mean, intensity by color gradient
        cs              = np.array(np.where(self.cat_all == cat)) # find indeces of specific cat
        cs              = np.reshape(cs, (-1))
        StimChan        = self.StimChans[sc]
        try: # load LL, calculate LL if non existing
            LL_SP = pd.read_csv(
                self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan +'_'+str(w)+'s.csv')
        except IOError:
            self.get_LL_sc(sc=sc, protocols=[0, 2], w=w)
            LL_SP = pd.read_csv(
                self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan + '_' + str(w) + 's.csv')
        # adding data from PP protocol
        LL_PP = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan + '_' + str(w) + 's.csv')
        LL_PP = LL_PP[LL_PP.IPI>w*1000]
        LL_PP.insert(3, "LL norm", 0, True)
        for c in range(len(self.labels_all)):
            mx                                          = np.max(LL_SP[(LL_SP['Chan'] == c)&(LL_SP['Condition'] == 'Baseline')]['LL'])
            LL_PP.loc[(LL_PP['Chan'] == c), 'LL norm']  = LL_PP[(LL_PP['Chan'] == c)]['LL SP'].values / mx
        if PP:
            LL_all =pd.concat([LL_SP, LL_PP], sort=False)
        else:
            LL_all = LL_SP
        fig = plt.figure(figsize=(18, 12))
        #sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        sns.set_palette(['#594157', "#8FB996"])
        g = sns.pointplot(x="Int", y="LL norm", hue="Condition", data=LL_all[LL_all['Chan'].isin(cs)], kind="point", ci="sd",
                      height=6,  # make the plot 5 units high
                      aspect=1.5, s=0.8, legend_out=False)
        for bar in g.patches:
            bar.set_zorder(3)

        #sns.catplot(x="Int", y="LL norm", hue="Condition", data=LL_all[LL_all['Chan'].isin(cs)],kind="boxen",
                    # height=6,  # make the plot 5 units high #s=4, swarm ,box
                    # aspect=1.5,  legend_out=False,
                    #   palette=sns.color_palette([self.color_elab[0, :], self.color_elab[2, :]]))
        #plt.xlabel('Int [mA]', fontsize=16)
        #plt.ylabel('LL norm ', fontsize=16)
        #plt.suptitle('LL ' + str(w) + 's --  Stim: ' + StimChan + ', Resp: ' + cat, fontsize=16)
        plt.tick_params(axis="both", labelsize=16)
        plt.legend(fontsize=16)

        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IO/IO_' + StimChan + '-' + cat[0:3] + '_' + str(
            w) + 's.jpg')
        plt.savefig(
            self.path_patient + '/Analysis/Pharmacology/LL/figures/IO/IO_' + StimChan + '-' + cat[0:3] + '_' + str(
                w) + 's.svg')

        plt.show()
    def IO_prot_cat(self, sc, cat, w):
        ### old
        #cs = [i for i, v in enumerate(self.labels_all) if cat in v]
        cs = np.array(np.where(self.cat_all == cat))
        cs = np.reshape(cs, (-1))
        #plot all mean, intensity by color gradient for multiple response channels
        StimChan        = self.StimChans[sc]
        try:
            LL_all          = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan +'_'+str(w)+'s.csv')
        except IOError:
            self.get_LL_sc(sc=sc, protocols=[0,2],w=w)
            LL_all = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan + '_' + str(w) + 's.csv')

        fig             = plt.figure(figsize=(10, 8))
        #plt.suptitle('LL '+str(w)+'s --  Stim: ' + StimChan + ', Resp: ' + self.labels_all[cs], fontsize=16)
        data            = LL_all[LL_all['Chan'].isin(cs)]
        data_LL         = np.zeros((1, 4)) # C, Int, condition
        # todo: find easier way
        for c in cs:
            for Int in np.unique(data.Int):
                for cond in np.unique(data.State):
                    val         = np.zeros((1, 4))
                    val[:, 0]   = c  # response channel
                    val[:, 1]   = Int
                    val[:, 2]   = np.nanmean(LL_all[(LL_all.Chan == c)&(LL_all.Int == Int)&(LL_all.State == cond)]['LL'])
                    val[:, 3]   = cond
                    data_LL     = np.concatenate((data_LL, val), axis=0)
        data_LL = data_LL[1:-1, :]
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Int": data_LL[:, 1], "LL": data_LL[:, 2],
             "State": data_LL[:, 3]})
        LL_all.insert(3, "Condition", np.repeat(self.t_label[0], LL_all.shape[0]), True)
        for p in range(len(self.t_all)):
            LL_all.loc[(LL_all.State == p), 'Condition'] = self.t_label[p]
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        sns.pointplot(x="Int", y="LL", hue="Condition", data=data, kind="point", ci="sd",
                      height=6,  # make the plot 5 units high
                      aspect=1.5, s=0.8, legend_out=False,
                      palette=sns.color_palette([self.color_elab[0, :], self.color_elab[2, :]]))
        plt.xlabel('Int [mA]', fontsize=16)
        plt.ylabel('LL [uV/ms]', fontsize=16)
        plt.suptitle('LL '+str(w)+'s --  Stim: ' + StimChan + ', Resp: '+cat, fontsize=16)
        plt.tick_params(axis="both", labelsize=16)
        plt.legend(fontsize=16)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IO/IO_prots_' + StimChan + '-'+cat[0:4]+ '_'+str(w)+'s.jpg')
        plt.show()
    def plot_trial(self, sc, c, Int, protocol):
        w       = 0.25
        # plot  mean for specific intensity, intensity by row
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        fig             = plt.figure(figsize=(8, 8))
        LL_all = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan + '_' + str(w) + 's.csv')
        t           = self.t_all[protocol]
        EEG_block   = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
        stim_table  = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
        stim_list_spef = stim_table[
            (stim_table.ChanP == ChanP) & (stim_table.Int_prob == Int) & (stim_table.noise == 0)]
        stimNum         = stim_list_spef.Number.values
        plt.suptitle(self.t_label[protocol]+', Stim: ' + StimChan + ', Resp: ' + self.labels_all[c] + ', Int: '+str(Int), fontsize=20)
        gs      = GridSpec(len(stimNum), 1)
        LL  = np.array(LL_all[(LL_all.Int==Int)&(LL_all.Chan==c)&(LL_all.State==protocol)]['LL'].values)
        LL_norm = np.array(LL_all[(LL_all.Int == Int) & (LL_all.Chan == c) & (LL_all.State == protocol)]['LL norm'].values)
        for i in range(len(stimNum)):
            resps           = EEG_block[c, stimNum[i], :]
            axs             = fig.add_subplot(gs[i, 0])
            axs.plot(self.x_ax, resps, linewidth=3, c=self.color_elab[protocol, :])
            plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
            plt.ylim([-700, 700])
            plt.xlim([-0.1, 2])
            plt.axvspan(0.02, 0.270, alpha=0.05, color='black')
            plt.xticks([])
            plt.yticks([])
            plt.title('LL: '+str(np.round(LL[i],2))+'uV/ms, Norm:'+str(np.round(LL_norm[i],2)))
                # plt.yticks(np.arange(-500,510,500))
                #plt.tick_params(axis="y", labelsize=14)
        plt.xticks(np.arange(0, 2.1, 0.5))
        plt.xlabel('time [s]', fontsize=20)
        plt.tick_params(axis="x", labelsize=18)
        plt.show()

    def plot_mean_grid(self, sc, c, protocols):
        # plot  mean for specific intensity, intensity by row
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        resps           = np.zeros((len(protocols), 3, 5000))
        fig             = plt.figure(figsize=(8, 8))
        plt.suptitle('Stim: ' + StimChan + ', Resp: ' + self.labels_all[c] + ', n=3', fontsize=20)
        int_resps       = [0.2, 0.4, 1, 2, 4, 6, 7, 8, 9, 10]  # int_resps = [0.2,1,2,4]
        gs              = GridSpec(len(int_resps), 1)
        for j in range(len(protocols)):
            t           = self.t_all[protocols[j]]
            EEG_block   = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table  = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            for i in range(len(int_resps)):
                Int         = int_resps[i]
                stim_list_spef = stim_table[
                    (stim_table.ChanP == ChanP) & (stim_table.Int_prob == Int) & (stim_table.noise == 0)]
                stimNum         = stim_list_spef.Number.values
                resps[j, :, :]  = EEG_block[c, stimNum, :]
                axs             = fig.add_subplot(gs[i, 0])
                axs.plot(self.x_ax, np.nanmean(resps[j, :, :], 0), linewidth=3, c=self.color_elab[protocols[j], :])
                axs.fill_between(self.x_ax, np.nanmean(resps[j, :, :], 0) - np.nanstd(resps[j, :, :], 0),
                                 np.nanmean(resps[j, :, :], 0) + np.nanstd(resps[j, :, :], 0),
                                 facecolor=self.color_elab[protocols[j], :], alpha=0.1)
                plt.ylabel(str(Int) + 'mA', fontsize=14)
                plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
                plt.ylim([-500, 500])
                plt.xlim([-0.1, 1])
                plt.axvspan(0.02, 0.12, alpha=0.05, color='black')
                plt.xticks([])
                plt.yticks([])
                # plt.yticks(np.arange(-500,510,500))
                #plt.tick_params(axis="y", labelsize=14)
        plt.xticks(np.arange(0, 1.1, 0.5))
        plt.xlabel('time [s]', fontsize=20)
        plt.tick_params(axis="x", labelsize=18)

        plt.show()

    def plot_mean_grid_cat(self, sc, c, protocols):
        # plot  mean for specific intensity, intensity by row
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        resps           = np.zeros((len(protocols), 3, 5000))
        fig             = plt.figure(figsize=(8, 8))
        plt.suptitle('Stim: ' + StimChan + ', Resp: ' + self.labels_all[c] + ', n=3', fontsize=20)
        int_resps       = [0.2, 0.4, 1, 2, 4, 6, 7, 8, 9, 10]  # int_resps = [0.2,1,2,4]
        gs              = GridSpec(len(int_resps), 1)
        for j in range(len(protocols)):
            t           = self.t_all[protocols[j]]
            EEG_block   = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table  = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            for i in range(len(int_resps)):
                Int         = int_resps[i]
                stim_list_spef = stim_table[
                    (stim_table.ChanP == ChanP) & (stim_table.Int_prob == Int) & (stim_table.noise == 0)]
                stimNum         = stim_list_spef.Number.values
                resps[j, :, :]  = EEG_block[c, stimNum, :]
                axs             = fig.add_subplot(gs[i, 0])
                axs.plot(self.x_ax, np.nanmean(resps[j, :, :], 0), linewidth=3, c=self.color_elab[protocols[j], :])
                axs.fill_between(self.x_ax, np.nanmean(resps[j, :, :], 0) - np.nanstd(resps[j, :, :], 0),
                                 np.nanmean(resps[j, :, :], 0) + np.nanstd(resps[j, :, :], 0),
                                 facecolor=self.color_elab[protocols[j], :], alpha=0.1)
                plt.ylabel(str(Int) + 'mA', fontsize=14)
                plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
                plt.ylim([-500, 500])
                plt.xlim([-0.1, 1])
                plt.axvspan(0.02, 0.12, alpha=0.05, color='black')
                plt.xticks([])
                plt.yticks([])
                # plt.yticks(np.arange(-500,510,500))
                #plt.tick_params(axis="y", labelsize=14)
        plt.xticks(np.arange(0, 1.1, 0.5))
        plt.xlabel('time [s]', fontsize=20)
        plt.tick_params(axis="x", labelsize=18)

        plt.show()

    def plot_PCA_IO(self, sc, exp):
        Int_all         = np.sort(self.Int_all)
        StimInd         = self.stim_ind[sc]
        badchan         = pd.read_excel(self.path_patient + '/Data/experiment' + str(exp) + '/' + self.subj + "_stimlist_Ph_IO.xlsx", sheet_name='BadChans')
        badchans            = np.concatenate([badchan.values[:, 0], [StimInd - 1, StimInd, StimInd + 1, StimInd + 2]])
        labels_all_     = np.delete(self.labels_all, badchans)
        StimChan        = self.StimChans[sc]
        print(StimChan)
        LL_all          = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_IO_' + StimChan + '.csv')
        pca,x_pca, labels_pca       = PCA_funcs.PCA_IO_mean(LL_all, badchans, [0,2], Int_all, self.labels_all, 4)
        # plot explained variances
        PCA_funcs.plot_variances(pca, self.color_elab, labels_pca)
        PCA_funcs.plot_PC_IO(x_pca, pca, Int_all, labels_pca)

        labels_all      = np.concatenate([labels_all_, labels_all_])
        loadings        = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_matrix  = pd.DataFrame(loadings, columns=labels_pca, index=labels_all)
        loading_matrix.insert(4, "State", np.repeat(self.t_label[0], loadings.shape[0]), True)
        loading_matrix.loc[len(labels_all_):, 'State'] = self.t_label[2]

        # get loadings for each protocol
        l_BL            = loading_matrix[loading_matrix.State == 'Baseline']
        l_B             = loading_matrix[loading_matrix.State == 'Benzodiazepin']

    def plot_mean_grad(self, sc, c, protocols,w):
        ###paper
        #plot all mean, intensity by color gradient
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        resps           = np.zeros((len(protocols), 3, 5000))

        for j in range(len(protocols)):
            t           = self.t_all[protocols[j]]
            #plt.figure(figsize=(12, 8))
            # fig     = plt.figure(figsize=(len(self.Int_all)*3,len(protocols)*2))
            fig = plt.figure(figsize=(9, 6))
            plt.suptitle('Stim: ' + StimChan + ', Resp: ' + self.labels_all[c] + ', '+self.t_label[protocols[j]], fontsize=20)
            EEG_block   = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table  = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            for i in range(len(self.Int_all)):
                Int             = self.Int_all[i]
                stim_list_spef  = stim_table[
                    (stim_table.ChanP == ChanP) & (stim_table.Int_prob == Int) & (stim_table.noise == 0)]
                stimNum         = stim_list_spef.Number.values
                #resps[j, :, :]  = EEG_block[c, stimNum, :]
                plt.plot(self.x_ax, np.nanmean(EEG_block[c, stimNum, :], 0), linewidth=3, c=self.colors_Int[i,:])

                plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=1.2)
                plt.ylim([-550, 550])
                plt.xlim([-0.5, 1])

                # plt.yticks(np.arange(-500,510,500))
                #plt.tick_params(axis="y", labelsize=14)
            plt.xticks(np.arange(-0.5, 1.1, 0.5))
            plt.yticks(np.arange(-300, 400, 300))
            plt.axvspan(0.02, w, alpha=0.05, color='black')
            plt.xlabel('time [s]', fontsize=18)
            plt.ylabel('uV', fontsize=18)
            plt.tick_params(axis="x", labelsize=14)
            plt.tick_params(axis="y", labelsize=12)
            plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IO/grad_' +self.t_label[protocols[j]]+'_'+ StimChan + '-' + str(
                self.labels_all[c]) + '_'+str(w)+'s.jpg')
            plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IO/grad_' + self.t_label[
                protocols[j]] + '_' + StimChan + '-' + str(
                self.labels_all[c]) + '_' + str(w) + 's.svg')
            plt.show()