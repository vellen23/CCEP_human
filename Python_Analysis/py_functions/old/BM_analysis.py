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
import LL_funcs
import freq_funcs
import plot_WT_funcs

from matplotlib.gridspec import GridSpec
import mne
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from math import sin
import dabest
import LL_funcs
from pandas import read_excel
from scipy.stats import norm

from scipy.io import savemat
import scipy.io


class main:
    def __init__(self,subj):
        if platform.system()=='Windows':
            sep = ','
        else: #'Darwin' for MAC
            sep =';'

        self.color_elab         = np.zeros((4, 3))
        self.color_elab[0, :]   = np.array([35, 25, 34]) / 255#np.array([31, 78, 121]) / 255
        self.color_elab[1, :]   = np.array([95, 75, 60]) / 255
        self.color_elab[2, :]   = np.array([56, 73, 59]) / 255#np.array([0.256, 0.574, 0.431])
        self.color_elab[3, :] = np.array([0.81, 0.33, 0.23])

        self.color_elab = ['#594157', "#F1BF98","#8FB996"]

        self.subj           = subj
        cwd                 = os.getcwd()
        self.path_patient   = os.path.dirname(os.path.dirname(cwd)) + '/Patients/' + subj
        labels_all          = pd.read_csv(self.path_patient + "/infos/" + subj + "_BP_labels.csv", header=0, dtype=str, sep=sep)
        self.labels_all     = labels_all.label.values
        self.cat_all        = labels_all.Cat.values
        data                = pd.read_csv(self.path_patient + "/infos/" + subj + "_BP_labels.csv", header=0, sep=sep)
        file_name           = subj + '_lookup.xlsx'
        #df                  = pd.read_excel(os.path.join(self.path_patient + "/infos/", file_name),sheet_name='Par_benzo')  # ChanP  ChanN  Int [mA]  IPI [ms]  ISI [s]  Num_P
        df = pd.read_excel(os.path.join(self.path_patient + "/infos/", file_name),
                           sheet_name='Par_Ph')  # ChanP  ChanN  Int [mA]  IPI [ms]  ISI [s]  Num_P
        stim_chan           = np.array(df.values[:, 0:2], dtype=float)
        stim_chan           = stim_chan[~np.isnan(stim_chan)]
        stim_chan           = stim_chan[np.nonzero(stim_chan)].reshape(-1, 2)
        self.StimChanNums   = stim_chan[:, 0]

        self.StimChans = []  # np.zeros((len(stim_chan)))
        for i in range(len(stim_chan)):
            self.StimChans.append(self.labels_all[(np.array(data.chan_BP_P.values) == stim_chan[i, 0]) & (np.array(data.chan_BP_N.values) == stim_chan[i, 1])][0])

        self.stim_ind = np.where(np.in1d(self.labels_all, self.StimChans))[0]

        badchan = pd.read_excel(self.path_patient + '/Data/experiment3/' + self.subj + "_stimlist_Ph_IO.xlsx",
            sheet_name='BadChans')
        self.badchans = np.concatenate([badchan.values[:, 0]-1]) #minus one because matlab -> python

        #self.IPI_all = df.IPI.values
        IPI_all             = df.IPI.values
        self.IPI_all        = np.sort(IPI_all[~np.isnan(IPI_all)])
        Int_all             = df.Int_cond.values
        self.Int_all        = Int_all[~np.isnan(Int_all)]

        self.Fs         = 500 #todo: remove hardcode
        self.dur        = np.zeros((1, 2), dtype=np.int)
        self.dur[0, :]  = [-5, 5]
        self.dur_tot    = np.int(np.sum(abs(self.dur)))
        self.x_ax       = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        self.colors_IPI         = np.zeros((len(self.IPI_all), 3))
        self.colors_IPI[:, 0]   = np.linspace(0, 1, len(self.IPI_all))

        self.t_all      = ['CR1', 'CR2', 'CR3']  # ['IO1', 'IO2', 'IO3']
        self.t_CR       = ['CR1', 'CR2', 'CR3']
        self.t_IO       = ['IO1', 'IO2', 'IO3']
        self.t_label    = ['Baseline', 'Flumazenil', 'Benzodiazepin']  # ['Baseline', 'Flumazenil', 'Benzodiazepin']

        self.f_bands        = np.array([[0.5,4],[4,8],[8,15],[15,30],[30,80],[80,200]])
        self.f_bands_label  = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'h_gammma']

    def cut_blocks_prot(self, exp, protocols):
        for i in protocols: #which protocols, e.g. protocols=[0,2], only IO1 and IO3 resp BL and Benzo
            t           = self.t_all[i]
            stim_table = pd.read_excel(self.path_patient + '/Data/experiment' + str(exp) + '/' + self.subj + "_stimlist_Ph_CR.xlsx",
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
                #EEG_block[:, s, :] = EEGpp[:, np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]
                if np.int(trig + self.dur[0, 1] * self.Fs) > EEGpp.shape[1]:
                    EEG_block[:, s, 0:EEGpp.shape[1] - np.int(trig + self.dur[0, 0] * self.Fs)] = EEGpp[:,
                                                                                        np.int(trig + self.dur[0, 0] * self.Fs):
                                                                                        EEGpp.shape[1]]
                elif np.int(trig + self.dur[0, 0] * self.Fs) < 0:
                    EEG_block[:, s, abs(np.int(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:, 0:np.int(trig + self.dur[0, 1] * self.Fs)]
                else:
                    EEG_block[:, s, :] = EEGpp[:, np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]

            stim_table.to_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv',index=False, header=True)
            np.save(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy', EEG_block)
            print('Data block saved \n')

    def cut_blocks_stim(self, sc, exp, protocols):

        dur        = np.zeros((1, 2), dtype=np.int)
        dur[0, :]  = [-1, 3]
        dur_tot    = np.int(np.sum(abs(dur)))


        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        stim_list   = []
        EEG         = np.zeros((len(self.labels_all), dur_tot * self.Fs)) # channels, 10s
        TTL         = np.zeros((1, dur_tot * self.Fs))  # channels, 10s
        for i in protocols: #which protocols, e.g. protocols=[0,2], only IO1 and IO3 resp BL and Benzo
            for q in [0,1]:
                if q == 0:
                    t = self.t_IO[i]
                    stim_table = pd.read_excel(
                        self.path_patient + '/Data/experiment' + str(exp) + '/' + self.subj + "_stimlist_Ph_IO.xlsx",
                        sheet_name=str(i + 1))
                else:
                    t = self.t_CR[i]
                    stim_table = pd.read_excel(
                        self.path_patient + '/Data/experiment' + str(exp) + '/' + self.subj + "_stimlist_Ph_CR.xlsx",
                        sheet_name=str(i + 1))

                stim_table.insert(0, "Number", np.arange(len(stim_table)), True)
                stim_table      = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)]
                EEG_block           = np.zeros((len(self.labels_all), len(stim_table)* dur_tot * self.Fs))
                TTL_block           = np.zeros((1, len(stim_table) * dur_tot * self.Fs))
                #EEG_block[:, :]  = np.NaN
                print(t)
                matfile = h5py.File(self.path_patient + '/Data/experiment' + str(exp) + '/data_blocks/time/' +self.subj + "_BP_Ph_" + t + "_pp/"+ self.subj + "_BP_Ph_" + t + "_pp.mat",
                          'r')['EEGpp']
                EEGpp = matfile[()].T
                for s in range(len(stim_table)):
                    trig = stim_table.TTL_DS.values[s]
                    EEG_block[:, s * dur_tot * self.Fs:(s+1)*dur_tot * self.Fs] = EEGpp[:, np.int(trig + dur[0, 0] * self.Fs):np.int(trig + dur[0, 1] * self.Fs)]
                    TTL_block[0, np.int(s * (-1) * dur[0, 0] * self.Fs)] = stim_table.Int_cond.values[s]
                    TTL_block[0, np.int(s * ((-1) * dur[0, 0]+stim_table.IPI_ms.values[s]/1000) * self.Fs)] = stim_table.Int_cond.values[s]
                    # #EEG_block[:, s, :] = EEGpp[:, np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]
                    # if np.int(trig + self.dur[0, 1] * self.Fs) > EEGpp.shape[1]:
                    #     EEG_block[:, s, 0:EEGpp.shape[1] - np.int(trig + self.dur[0, 0] * self.Fs)] = EEGpp[:,
                    #                                                                         np.int(trig + self.dur[0, 0] * self.Fs):
                    #                                                                         EEGpp.shape[1]]
                    # elif np.int(trig + self.dur[0, 0] * self.Fs) < 0:
                    #     EEG_block[:, s, abs(np.int(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:, 0:np.int(trig + self.dur[0, 1] * self.Fs)]
                    # else:
                    #     EEG_block[:, s*self.dur_tot * self.Fs:(s+1)*self.dur_tot * self.Fs] = EEGpp[:, np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]
                if i == 0 and q ==0:
                    stim_list = stim_table
                else:
                    stim_list = pd.concat([stim_list, stim_table], sort=False)
                EEG         = np.concatenate([EEG, EEG_block], 1)
                TTL         = np.concatenate([TTL, TTL_block], 1)

        EEG = {"EEG": EEG, "Fs": self.Fs}
        TTL = {"TTL": TTL, "Fs": self.Fs}
        savemat(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/Stim_' + StimChan + '.mat', EEG)
        savemat(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan + '/TTL.mat',TTL)
        stim_list.to_csv(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/Stim_' + StimChan + '_stimtable.csv', index=False,
                          header=True)
        print('Data block saved \n')

    def get_NNMF_coef(self, sc,protocols):
        StimChan = self.StimChans[sc]
        ChanP = self.StimChanNums[sc]

        dur         = np.zeros((1, 2), dtype=np.int)
        dur[0, :]   = [-1, 3]
        dur_tot     = np.int(np.sum(abs(dur)))
        stim_table  = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/Stim_' + StimChan + '_stimtable.csv')
        #TTL         = scipy.io.loadmat(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan + '/TTL.mat')['TTL']
        #EEG         = scipy.io.loadmat(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/Stim_' + StimChan + '.mat')['EEG']
        mat         = h5py.File(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/NNMF.mat','r')['NNMF']['H']
        NNMF_H      = mat[()].T
        #NNMF_H      = NNMF_H[H-1,:] # coefficients
        mat         = h5py.File(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan + '/NNMF.mat', 'r')[
            'NNMF']['W'] # features, weights of channels
        NNMF_W      = mat[()].T
        #chan_resp   = np.where(NNMF_W[:,H-1]>1000)

        w               = 0.1
        data_coef       = np.zeros((1, 7))  # RespChan, Int, LL, LLnorm, State
        self.SP_LL_mean = np.zeros((len(self.labels_all), len(self.t_label)))

        #IPIs            = np.array(stim_table.IPI_ms.values)
        coef            = LL_funcs.get_NNMFcoeff_both(data=NNMF_H, Fs=50, stim_list = stim_table, t_0=1, win=w)  # both: SP and PP

        for h in range(len(NNMF_H)):
            val       = np.zeros((coef.shape[1], 7))
            val[:, 0] = h+1  # response channel
            val[:, 1] = stim_table.Int_cond.values  # probing intensity
            val[:, 2] = stim_table.Int_prob.values  # probing intensity
            val[:, 3] = stim_table.IPI_ms.values  # IPI
            val[:, 4] = coef[h, :, 0]  # absolut value of first response
            val[:, 5] = coef[h, :, 1]  # absolut value of second response
            val[:, 6] = stim_table.stim_block.values  # protocol number

            data_coef = np.concatenate((data_coef, val), axis=0)

        data_coef = data_coef[1:-1, :]
        NNMF_coef = pd.DataFrame(
            {"H": data_coef[:, 0], "Int": data_coef[:, 1], "Int_prob": data_coef[:, 2], "IPI": data_coef[:, 3], "coeff SP": data_coef[:, 4],
             "coeff PP": data_coef[:, 5],
             "State": data_coef[:, 6]})
        NNMF_coef.loc[NNMF_coef['IPI'] == 0, 'coeff SP'] = NNMF_coef[NNMF_coef['IPI'] == 0]['coeff PP'].values
        NNMF_coef.loc[NNMF_coef['IPI'] == 0, 'Int'] = NNMF_coef[NNMF_coef['IPI'] == 0]['Int_prob'].values
        NNMF_coef.insert(6, "Condition", np.repeat(self.t_label[0], NNMF_coef.shape[0]), True)
        for j in protocols:
            NNMF_coef.loc[(NNMF_coef.State == j), 'Condition'] = self.t_label[j]

        #NNMF_coef.drop('State', axis='columns', inplace=True)
        NNMF_coef.drop('Int_prob', axis='columns', inplace=True)
        NNMF_coef.to_csv(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/NNMF_coeff' + '.csv', index=False,
                      header=True)  # scat_plot = scat_plot.fillna(method='ffill')
        print('Data saved')

    def plot_NNMF_IO(self, sc, H):
        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        #try:
        NNMF_coef   = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/NNMF_coeff' + '.csv')
        #except IOError:
         #   self.get_NNMF_coef
        #   NNMF_coef   = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/NNMF_coeff' + '.csv')

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])

        grid = sns.catplot(x="Int", y='coeff SP', hue="Condition", data=NNMF_coef[NNMF_coef['H']==H], kind="point",
                    height=3.5,  # make the plot 5 units high
                    aspect=3, s=5, legend_out=True)

        #plt.ylim([-0.5, 1.5])
        #plt.xticks(np.arange(6), ['H1', 'H2', 'H3', 'H4'])
        plt.xlabel('Int [mA]', fontsize=18)
        plt.ylabel('coeff [0.1s]', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.suptitle(str(self.subj + "-- SP Stim: " + StimChan + ', NNMF coefficients: '+str(H) ),
                                  fontsize=20)

        plt.savefig(self.path_patient + '/Analysis/Pharmacology/NNMF/figures/NNMF_IO_'+ StimChan + '_'+str(H)+'.jpg')

    def plot_NNMF_time(self, sc, c, protocols):
        dur         = np.zeros((1, 2), dtype=np.int)
        dur[0, :]   = [-1, 3]
        dur_tot     = np.int(np.sum(abs(dur)))
        Fs = 50
        ax          = np.arange(-1,3,1/Fs)
        ax_eeg = np.arange(-1,3,1/self.Fs)

        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        mat = \
        h5py.File(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan + '/NNMF.mat', 'r')[
            'NNMF']['H']
        NNMF_H      = mat[()].T
        stim_table  = pd.read_csv(
            self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan + '/Stim_' + StimChan + '_stimtable.csv')
        EEG         = scipy.io.loadmat(
            self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan + '/Stim_' + StimChan + '.mat')[
            'EEG']
        Ints = [1,2,4]
        plt.figure(figsize=(6*len(Ints), 2.5*len(NNMF_H)))
        for i in protocols:
            for j in range(len(Ints)):
                Int = Ints[j]
                stim_list_spef  = stim_table[(stim_table.stim_block==i)&(stim_table.Int_prob == Int)&(stim_table.IPI_ms ==0)]
                stimNum         = stim_list_spef.Number.values
                eeg_spec        = np.zeros((len(stimNum), dur_tot*self.Fs))
                for l in range(len(stimNum)):
                    eeg_spec[l, :] = EEG[c, np.int((stimNum[l] + 1) * 4 * self.Fs): np.int((stimNum[l] + 2) * 4 * self.Fs)]
                for k in range(len(NNMF_H)):
                    data = np.zeros((len(stimNum), dur_tot*Fs))
                    for l in range(len(stimNum)):
                        data[l,:] = NNMF_H[k,np.int((stimNum[l]+1)*4*Fs): np.int((stimNum[l]+2)*4*Fs)]

                    mn              = np.nanmean(data, axis=0)
                    std             = np.nanstd(data, axis=0)
                    plt.subplot(len(NNMF_H)+1, len(Ints), k*len(Ints)+j+len(Ints)+1)
                    plt.plot(ax,mn, c=self.color_elab[i, :])
                    plt.fill_between(ax, mn - std,
                                 mn + std,facecolor=self.color_elab[i, :], alpha=0.1)
                    plt.ylabel('H'+str(k+1))
                    plt.xlabel('')
                    plt.xlim([-0.1, 1])
                    plt.ylim([0, 0.04])
                    plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
                    plt.xticks([])
                    if k == len(NNMF_H)-1:
                        plt.xticks([0, 0.5, 1])
                        plt.xlabel('time [s]')

                plt.subplot(len(NNMF_H) + 1, len(Ints), j + 1)
                plt.title('Int: '+str(Int)+'mA')
                plt.plot(ax_eeg, np.nanmean(eeg_spec, axis=0), c=self.color_elab[i, :], label=self.t_label[i])
                plt.ylabel('uV')
                plt.xlabel('time [s]')
                plt.xlim([-0.1, 1])
                plt.ylim([-800, 800])
                plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
                plt.xticks([])
            plt.legend()
        plt.suptitle(self.subj + ' -- NNMF coeffcients, SP-Stims, '+StimChan, fontsize=18)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/NNMF/figures/NNMF_all_' + StimChan + '.jpg')
    def plot_NNMF_H_SP(self, sc):
        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        NNMF_coef   = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/NNMF_coeff' + '.csv')

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])

        grid = sns.catplot(x="H", y='coeff SP', hue="Condition", data=NNMF_coef[NNMF_coef['Int'].isin([1,2,4])], row='Int', kind="swarm",
                    height=3.5,  # make the plot 5 units high
                    aspect=3, s=5, legend_out=True)

        #plt.ylim([-0.5, 1.5])
        plt.xticks(np.arange(4), ['H1', 'H2', 'H3', 'H4'])
        #plt.xticks(np.arange(3), ['H1', 'H2', 'H3'])
        plt.xlabel('NNMF H number', fontsize=18)
        grid.axes[0, 0].set_title(str(self.subj + "-- SP Stim: " + StimChan + ', NNMF coefficients ' ),
                                  fontsize=20)
        for i in range(len(self.Int_all)):
            grid.axes[i, 0].set_ylabel('Mean coeff [0.1s]', fontsize=18)
            grid.axes[i, 0].tick_params(axis='y', labelsize=10)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/NNMF/figures/NNMF_H_SP_'+ StimChan + '.jpg')

    def plot_NNMF_H_PP(self, sc, Int, H):
        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        NNMF_coef   = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/Stim_block/Stim_' + StimChan +'/NNMF_coeff' + '.csv')

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])

        num_IPI     = len(np.unique(np.sort(NNMF_coef['IPI'])))
        colors_IPI = np.zeros((num_IPI, 3))
        colors_IPI[:, 0] = np.linspace(0, 1, num_IPI)

        #fig = plt.figure(figsize=(10, 15))
        sns.catplot(x="Condition", y='coeff PP', data=NNMF_coef[(NNMF_coef.Int==Int)&(NNMF_coef.H==H)], hue='IPI',palette=sns.color_palette(colors_IPI), kind='swarm',
                    height=7,  # make the plot 5 units high
                    aspect=1, s=5, legend_out=True)
        #plt.ylim([-0.5, 1.5])
        plt.suptitle(self.subj + ' -- NNMF coeffcients of H'+str(H)+', PP-Stims, Cond. Int: '+str(Int), fontsize=18)
        plt.subplots_adjust(top=0.85)
        plt.xlabel('Condition', fontsize=15)
        plt.ylabel('NNMF H coefficient [0.1s]', fontsize=15)
        plt.tick_params(axis='both', labelsize=13)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/NNMF/figures/NNMF_H' +str(H)+'_PP_Int'+str(Int)+ StimChan + '.jpg')

    def get_LL_bL(self, protocols):
        sc          = 0
        w           = 0.1
        data_LL     = np.zeros((1, 2))  # RespChan, Int, LL, LLnorm, State
        self.SP_LL_mean = np.zeros((len(self.labels_all), len(self.t_label)))
        j = 0
        t = self.t_all[j]
        EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
        stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
        stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)]
        stimNum         = stim_list_spef.Number.values

        IPIs            = np.array(np.zeros_(stim_list_spef.IPI_ms.values))
        LL = LL_funcs.get_LL_both(data=EEG_block[:, stimNum, :], Fs=500, IPI=IPIs, t_0=3, win=w)  # both: SP and PP

        for c in range(len(LL)):
            val         = np.zeros((LL.shape[1], 2))
            val[:, 0]   = c  # response channel
            val[:, 1]   = LL[c, :, 1]  # absolut value of first response

            data_LL = np.concatenate((data_LL, val), axis=0)

        data_LL = data_LL[1:-1, :]
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "LL": data_LL[:, 1]})

        self.LL_BL       = np.zeros((len(self.labels_all, 3)))
        for i in range(len(self.labels_all)):
            self.LL_BL[i, 0]         = np.nanmean(LL_all[LL_all.Chan==c]['LL'].values)
            self.LL_BL[i, 1]         = np.nanstd(LL_all[LL_all.Chan == c]['LL'].values)
            self.LL_BL[i, 2]         = np.percentile(LL_all[LL_all.Chan == c]['LL'].values, 99)

    def plot_resp_IPI_c(self, sc, c):
        w           = 0.1
        StimChan    = self.StimChans[sc]
        print(StimChan)
        print(self.labels_all[c])
        LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_CR_' + StimChan + '.csv')
        LL_all.insert(9, 'Resp', 0)
        LL_all.loc[(LL_all['Chan'] == c) & (LL_all['Condition'] == 'Benzodiazepin')& (LL_all['LL PP'] >self.LL_BL[c,2]), 'Resp'] = 1

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        data        = LL_all[(LL_all['IPI'] > 100) & (LL_all['Int'] > 0) & (LL_all['Chan'] == c)]
        mx          = np.max(data['LL PP norm Cond'])
        grid        = sns.catplot(x="IPI", y='Resp', hue="Condition", data=data, row='Int', kind="swarm",
                    height=3.5,  # make the plot 5 units high
                    aspect=1.8, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
        plt.ylim([-0.5, 1.5])
        plt.xticks([0, 2, 5, 7, 9,11, 13, 15], ['100', '150', '250', '350', '500', '700', '1000', '1600'])
        plt.xlabel('IPI [ms]', fontsize=18)
        plt.ylabel('LL norm Cond', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.axhline(1, ls='--', c=self.color_elab[0, :], linewidth=1.5)
        plt.axhline(self.SP_LL_mean[c, 2] / self.SP_LL_mean[c, 0], ls='--', c=self.color_elab[2, :],
                    linewidth=1.5)

        grid.axes[0, 0].set_title(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + str(self.labels_all[c])), fontsize=20)
        grid.axes[0, 0].set_ylabel('LL norm Cond', fontsize=18)
        grid.axes[0, 0].tick_params(axis='y', labelsize=14)
        grid.axes[0, 0].axhline(1, ls='--', c=self.color_elab[0, :], linewidth=1.5)
        grid.axes[0, 0].axhline(self.SP_LL_mean[c, 2] / self.SP_LL_mean[c, 0], ls='--', c=self.color_elab[2, :],
                    linewidth=1.5)
        grid.axes[1, 0].set_ylabel('LL norm Cond', fontsize=18)
        grid.axes[1, 0].tick_params(axis='y', labelsize=14)
        grid.axes[1, 0].axhline(1, ls='--', c=self.color_elab[0, :], linewidth=1.5)
        grid.axes[1, 0].axhline(self.SP_LL_mean[c, 2] / self.SP_LL_mean[c, 0], ls='--', c=self.color_elab[2, :],
                    linewidth=1.5)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_normCond_IPI_'+StimChan+ '-'+str(self.labels_all[c])+'.jpg')
        #plt.show()

    def get_LL_sc(self, sc, protocols,w):
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        print(StimChan)
        data_LL         = np.zeros((1, 6))  # RespChan, Int, LL, LLnorm, State
        self.SP_LL_mean      = np.zeros((len(self.labels_all),len(self.t_label))) #get mean 2mA LL for each channel and each protocol
        for j in protocols:
            t               = self.t_all[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)]
            stimNum         = stim_list_spef.Number.values

            IPIs            = np.array(stim_list_spef.IPI_ms.values)
            LL              = LL_funcs.get_LL_both(data=EEG_block[:, stimNum, :], Fs=500, IPI=IPIs, t_0=5, win=w) #both: SP and PP

            for c in range(len(LL)): #for each channel
                val         = np.zeros((LL.shape[1], 6))
                val[:, 0]   = c #response channel
                val[:, 1]   = stim_list_spef.Int_cond.values #probing intensity
                val[:, 2]   = stim_list_spef.IPI_ms.values  # IPI
                val[:, 3]   = LL[c, :, 0]  # absolut value of first response
                val[:, 4]   = LL[c, :, 1]  # absolut value of second response
                val[:, 5]   = j # protocol number

                data_LL = np.concatenate((data_LL, val), axis=0)

            ##normalize LL by  mean SP 2mA
            stim_list_spef  = stim_table[((stim_table.ChanP == ChanP) & (stim_table.noise == 0) & (stim_table.Int_cond == 2) & (stim_table.IPI_ms > 1000*w)) | (
                                                    stim_table.ChanP == ChanP) & (stim_table.IPI_ms == 0) & (
                                                    stim_table.noise == 0)]
            stimNum         = stim_list_spef.Number.values
            IPIs            = np.zeros(stim_list_spef.IPI_ms.values.shape)
            LL_SP           = LL_funcs.get_LL_both(data=EEG_block[:, stimNum, :], Fs=500, IPI=IPIs, t_0=5,
                                         win=w)  # data_LL     = LL[0,:,0]
            self.SP_LL_mean[:, j] = np.nanmean(LL_SP[:, :, 1], axis=1)
            #data_LL = np.concatenate((data_LL, val), axis=0)

        data_LL = data_LL[1:-1, :]
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Int": data_LL[:, 1], "IPI": data_LL[:, 2], "LL SP": data_LL[:, 3], "LL PP": data_LL[:, 4],
             "State": data_LL[:, 5]})
        LL_all.insert(6, "Condition", np.repeat(self.t_label[0], LL_all.shape[0]), True)
        for j in protocols:
            LL_all.loc[(LL_all.State == j), 'Condition'] = self.t_label[j]

        LL_all.loc[LL_all['IPI'] == 0, 'LL SP'] = LL_all[LL_all['IPI'] == 0]['LL PP'].values
        # fill LL sp for PP with IPI < LL window
        for i in range(len(self.Int_all)):
            Int                             = self.Int_all[i]
            LL_all[LL_all['Int'] == Int]    = LL_all[LL_all['Int'] == Int].fillna(method='bfill', axis=0)
            LL_all[LL_all['Int'] == Int]    = LL_all[LL_all['Int'] == Int].fillna(method='ffill', axis=0)

        # l PP/SP ratio
        LL_all.insert(5, "LL PP/SP", (LL_all['LL PP'] / LL_all['LL SP']).values, True)
        # PP normalized by BL(mean 2mA SP)
        LL_all.insert(6, "LL SP norm BL", 0, True)
        # SP normalized by BL (mean 2mA SP)
        LL_all.insert(7, "LL PP norm BL", 0, True)
        # PP normalized by Cond (mean 2mA SP)
        LL_all.insert(8, "LL SP norm Cond", 0, True)
        # SP normalized by Cond (mean 2mA SP)
        LL_all.insert(9, "LL PP norm Cond", 0, True)
        for i in range(len(self.labels_all)):
            LL_all.loc[LL_all['Chan'] == i, 'LL SP norm BL'] = LL_all[LL_all['Chan'] == i]['LL SP'].values / self.SP_LL_mean[i, 0]
            LL_all.loc[LL_all['Chan'] == i, 'LL PP norm BL'] = LL_all[LL_all['Chan'] == i]['LL PP'].values / self.SP_LL_mean[i, 0]
            for j in protocols:
                LL_all.loc[(LL_all['Chan'] == i)&(LL_all['State'] == j), 'LL SP norm Cond'] = LL_all[(LL_all['Chan'] == i)&(LL_all['State'] == j)]['LL SP'].values / self.SP_LL_mean[i, j]
                LL_all.loc[(LL_all['Chan'] == i)&(LL_all['State'] == j), 'LL PP norm Cond'] = LL_all[(LL_all['Chan'] == i)&(LL_all['State'] == j)]['LL PP'].values / self.SP_LL_mean[i, j]

        #LL_all.drop('State',axis='columns', inplace=True)
        LL_all.to_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan +'_'+str(w)+'s.csv', index=False,
                      header=True)  # scat_plot = scat_plot.fillna(method='ffill')
        print('Data saved')
    def IO_prot_norm_cat(self, sc, cat, w):
        # plot LL normalized by max for all channels within a specific brain area (cat, category)
        ## GABA paper

        #plot all mean, intensity by color gradient
        cs              = np.array(np.where(self.cat_all == cat)) # find indeces of specific cat
        cs              = np.reshape(cs, (-1))
        StimChan        = self.StimChans[sc]
        try: # load LL, calculate LL if non existing
            LL_all = pd.read_csv(
                self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan +'_'+str(w)+'s.csv')
        except IOError:
            self.get_LL_sc(sc=sc, protocols=[0, 2], w=w)
            LL_all = pd.read_csv(
                self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan + '_' + str(w) + 's.csv')
        fig             = plt.figure(figsize=(18, 12))

        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        sns.pointplot(x="Int", y="LL norm", hue="Condition", data=LL_all[LL_all['Chan'].isin(cs)], kind="point", ci="sd",
                      height=6,  # make the plot 5 units high
                      aspect=1.5, s=0.8, legend_out=False,
                      palette=sns.color_palette([self.color_elab[0, :], self.color_elab[2, :]]))

        #sns.catplot(x="Int", y="LL norm", hue="Condition", data=LL_all[LL_all['Chan'].isin(cs)],kind="boxen",
                    # height=6,  # make the plot 5 units high #s=4, swarm ,box
                    # aspect=1.5,  legend_out=False,
                    #   palette=sns.color_palette([self.color_elab[0, :], self.color_elab[2, :]]))
        #plt.xlabel('Int [mA]', fontsize=16)
        #plt.ylabel('LL norm ', fontsize=16)
        #plt.suptitle('LL ' + str(w) + 's --  Stim: ' + StimChan + ', Resp: ' + cat, fontsize=16)
        plt.tick_params(axis="both", labelsize=16)
        plt.legend(fontsize=16)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IO/IO_norm_' + StimChan + '-' + cat[0:3] + '_' + str(
            w) + 's.jpg')
        plt.savefig(
            self.path_patient + '/Analysis/Pharmacology/LL/figures/IO/IO_norm_' + StimChan + '-' + cat[0:3] + '_' + str(
                w) + 's.svg')
        plt.show()
    def plot_LL_SP(self, sc, c, protocols, w=0.25):
        StimChan = self.StimChans[sc]
        #ChanP = self.StimChanNums[sc]
        print(StimChan)
        print(self.labels_all[c])
        LL_all  = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan +'_'+str(w)+'s.csv')
        LL_IO   = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan +'_'+str(w)+'s.csv')
        plt.style.use('seaborn-colorblind')
        fig     = plt.figure(figsize=(len(self.Int_all)*3,len(protocols)*2))
        plt.suptitle(str(self.subj + "--SP - LL [0.1s], Stim: " + StimChan + ', Resp: ' + self.labels_all[c]), fontsize=14, y=1)
        gs      = fig.add_gridspec(1,len(self.Int_all))  #fig.add_gridspec(len(self.Int_all), 1)# GridSpec(4,1, height_ratios=[1,2,1,2])
        mx      = np.max(LL_all[(LL_all['IPI'] > 1000*(w+0.02)) & (LL_all['Int'] ==4) & (LL_all['Chan'] == c)])['LL SP norm BL']
        for i in range(len(self.Int_all)):
            Int = self.Int_all[i]
            data_PP = LL_all[(LL_all['IPI'] > 1000*(w+0.02)) &(LL_all['Int'] ==Int) & (LL_all['Chan'] == c)]
            #data_SP = LL_IO[(LL_IO['Int'] ==Int) & (LL_IO['Chan'] == c)]
            #data     = pd.concat([data_PP, data_SP], sort=False)

            axs     = fig.add_subplot(gs[0,i]) #i,0
            sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
            sns.stripplot(ax=axs, x="Condition", y="LL SP norm BL", hue="Condition",
                        data=data_PP,size=8, jitter=0.2)
            sns.barplot(ax=axs, x="Condition", y="LL SP norm BL", hue="Condition", data=data_PP, ci="sd", saturation=.5)
            plt.legend([], [], frameon=False)
            plt.ylim([0, mx + 0.1])
            plt.title('Intensity: ' + str(Int) + 'mA', fontsize=10)
            plt.ylabel('', fontsize=10)
            plt.xlabel('', fontsize=10)
            plt.yticks([0, 0.5, 1, 1.5])
            plt.axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
            plt.xticks([0, 1], ['Baseline', 'Benzodiazepine'])
            plt.tick_params(axis='both', labelsize=10)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/SP/LL_Int_SP_bar_' + StimChan + '-' + str(self.labels_all[c]) + '.jpg')
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/SP/LL_Int_SP_bar_' + StimChan + '-' + str(
            self.labels_all[c]) + '.svg')
        #plt.show()
    def plot_LL_SP_cat(self, sc, cat, protocols, w=0.1, IO=False):
        ## for paper
        # plot LL normalized by 2mA baseline for all channels within an area (categorie, cat)
        # todo: adding bootstrp
        StimChan = self.StimChans[sc]
        #ChanP = self.StimChanNums[sc]
        print(StimChan)

        cs = np.array(np.where(self.cat_all == cat))  # find indeces of specific cat
        cs = np.reshape(cs, (-1))
        try:  # load LL, calculate LL if non existing
            LL_all = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan +'_'+str(w)+'s.csv')
        except IOError:
            self.get_LL_sc(sc=sc, protocols=protocols, w=w)
            LL_all = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan +'_'+str(w)+'s.csv')
        LL_IO = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan +'_'+str(w)+'s.csv')
        plt.style.use('seaborn-colorblind')
        #fig     = plt.figure(figsize=(len(self.Int_all)*3,len(protocols)*2))
        #fig = plt.figure(figsize=(len(self.Int_all) * 4, len(self.Int_all) * 4))
        fig = plt.figure(figsize=(18, 6))
        plt.suptitle('SP LL ' + str(w) + 's norm--  Stim: ' + StimChan + ', Resp: ' + cat, fontsize=16, y=1)
        gs      = fig.add_gridspec(1,len(self.Int_all))  #fig.add_gridspec(len(self.Int_all), 1)# GridSpec(4,1, height_ratios=[1,2,1,2])
        mx      = np.max(LL_all[(LL_all['IPI'] > 1000*w) & (LL_all['Int'] == 4) & (LL_all['Chan'].isin(cs))])['LL SP norm BL']

        ###both protocols
        data_PP     = LL_all[(LL_all['IPI'] > 1000 * w) & (LL_all['Chan'].isin(cs))]
        # num             = len(LL_all[(LL_all['IPI'] > 1000*w) &(LL_all['Int'] ==Int) & (LL_all['Chan']==cs[0])]['LL'].values)
        data_SP     = LL_IO[LL_IO['Chan'].isin(cs)]
        data_SP.insert(3, "LL SP normed", 0, True)
        data_SP.insert(3, "LL SP", data_SP['LL'], True)
        data_PP.insert(3, "LL SP normed", 0, True)
        if IO:
            data_PP = pd.concat([data_SP, data_PP], sort=False)
        for c in cs:
            ref = np.mean(data_PP[(data_PP['Chan'] == c) & (data_PP['Condition'] == 'Baseline') & (data_PP['Int'] == 2)]['LL SP'])
            # ref = np.mean(data_PP[(data_PP['Chan'] == c) & (data_PP['Condition'] == 'Baseline')]['LL SP']/data_PP[(data_PP['Chan'] == c) & (data_PP['Condition'] == 'Baseline')]['LL SP norm BL'])
            data_PP.loc[(data_PP['Chan'] == c), 'LL SP normed'] = data_PP[(data_PP['Chan'] == c)]['LL SP'].values / ref
        for i in range(len(self.Int_all)):
            Int             = self.Int_all[i]
            #data     = pd.concat([data_PP, data_SP], sort=False)
            axs     = fig.add_subplot(gs[0,i]) #i,0
            #sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
            sns.set_palette(['#594157', "#8FB996"])
            sns.stripplot(ax=axs, x="Condition", y="LL SP normed", hue="Condition",
                        data=data_PP[data_PP.Int==Int],size=8, jitter=0.2)
            sns.barplot(ax=axs, x="Condition", y="LL SP normed", hue="Condition", data=data_PP[data_PP.Int==Int], ci="sd", saturation=.5)
            plt.legend([],[], frameon=False)
            plt.ylim([0, mx+0.1])
            #plt.title('Intensity: '+str(Int)+'mA', fontsize=10)
            plt.ylabel('', fontsize=10)
            plt.xlabel('', fontsize=10)
            plt.yticks([0, 0.5, 1, 1.5], ['', '', '', ''])
            plt.axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
            #plt.xticks([0,1],['Baseline','Benzodiazepine'])
            plt.xticks([0, 1], ['', ''])
            plt.tick_params(axis='both', labelsize=10)

        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/SP/LL_SP_bar_' + StimChan + '-' + cat[0:3] + '_' + str(w) + 's.jpg')
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/SP/LL_SP_bar_' + StimChan + '-' + cat[0:3] + '_' + str(w) + 's.svg')

        #plt.show()
    def plot_LL_SP_cat_BS(self, sc, cat, protocols, w=0.1):
        sns.set_palette(['#594157', "#8FB996"])
        ## for paper
        # plot LL normalized by 2mA baseline for all channels within an area (categorie, cat)
        # todo: adding bootstrp
        StimChan = self.StimChans[sc]
        #ChanP = self.StimChanNums[sc]
        print(StimChan)
        cs = np.array(np.where(self.cat_all == cat))  # find indeces of specific cat
        cs = np.reshape(cs, (-1))
        try:  # load LL, calculate LL if non existing
            LL_all = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan +'_'+str(w)+'s.csv')
        except IOError:
            self.get_LL_sc(sc=sc, protocols=protocols, w=w)
            LL_all = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan +'_'+str(w)+'s.csv')
        plt.style.use('seaborn-colorblind')
        #fig     = plt.figure(figsize=(len(self.Int_all)*3,len(protocols)*2))
        #fig = plt.figure(figsize=(len(self.Int_all) * 4, len(self.Int_all) * 4))
        fig = plt.figure(figsize=(18, 9))
        plt.suptitle('SP LL ' + str(w) + 's norm--  Stim: ' + StimChan + ', Resp: ' + cat, fontsize=16, y=1)
        gs      = fig.add_gridspec(1,len(self.Int_all))  #fig.add_gridspec(len(self.Int_all), 1)# GridSpec(4,1, height_ratios=[1,2,1,2])
        mx      = np.max(LL_all[(LL_all['IPI'] > 1000*w) & (LL_all['Int'] == 4) & (LL_all['Chan'].isin(cs))])['LL SP norm BL']

        ###both protocols
        data_PP     = LL_all[(LL_all['IPI'] > 1000 * w) & (LL_all['Chan'].isin(cs))]
        data_PP.insert(3, "LL SP normed", 0, True)
        for c in cs:
            ref = np.mean(data_PP[(data_PP['Chan'] == c) & (data_PP['Condition'] == 'Baseline') & (data_PP['Int'] == 2)]['LL SP'])
            data_PP.loc[(data_PP['Chan'] == c), 'LL SP normed'] = data_PP[(data_PP['Chan'] == c)]['LL SP'].values / ref
        for i in range(len(self.Int_all)):
            Int             = self.Int_all[i]
            #data
            data_LL = data_PP[(data_PP['Int'] == Int)]
            # data = pd.DataFrame({'Baseline': data_LL[data_LL.State == 0]['LL SP normed'].values[0:36],
            #                      'Benzodiazepine': data_LL[data_LL.State == 2]['LL SP normed'].values[0:36]})
            data = pd.DataFrame({'Baseline': data_LL[data_LL.State == 0]['LL SP normed'],
                                 'Benzodiazepine': data_LL[data_LL.State == 2]['LL SP normed']})
            two_groups_unpaired = dabest.load(data, idx=("Baseline", "Benzodiazepine"), ci=95, resamples=5000)

            #data     = pd.concat([data_PP, data_SP], sort=False)
            axs             = fig.add_subplot(gs[0,i]) #i,0

            sns.set_palette(['#594157', "#8FB996"])
            sns.barplot(ax=axs, x="Condition", y="LL SP normed", hue="Condition", data=data_LL,
                        ci="sd", saturation=.5)
            plt.ylabel('', fontsize=10)
            plt.xlabel('', fontsize=10)
            plt.yticks([0, 0.5, 1, 1.5], ['', '', '', ''])
            plt.legend([], [], frameon=False)
            two_groups_unpaired.mean_diff.plot(ax=axs, raw_marker_size=6, swarm_label='Normalized Line Length',
                                               swarm_ylim=[0, mx + 0.1], contrast_ylim=[-0.3, 0.1],
                                               custom_palette=['#594157', "#8FB996"],float_contrast=False)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/SP/LL_SP_BS_' + StimChan + '-' + cat[0:3] + '_' + str(w) + 's.jpg')
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/SP/LL_SP_BS_' + StimChan + '-' + cat[0:3] + '_' + str(w) + 's.svg')

        #plt.show()
    def plot_SP(self, sc, c,  protocols, w=0.1):
        ## for paper GABA
        # plot SP response for all of the coniditoining intensities (portocol overlaied)
        lim         = 0.5
        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        print(StimChan)
        print(self.labels_all[c])
        # fig     = plt.figure(figsize=(len(self.Int_all)*3,len(protocols)*2))
        #fig = plt.figure(figsize=(len(self.Int_all) * 4, len(self.Int_all) * 4))
        fig = plt.figure(figsize=(18,6))
        gs = fig.add_gridspec(1, len(self.Int_all))
        plt.suptitle(str(self.subj + "--SP - LL [0.1s], Stim: " + StimChan + ', Resp: ' + self.labels_all[c]), fontsize=14, y=1)
        for k in range(len(protocols)):
            for i in range(len(self.Int_all)):
                Int = self.Int_all[i]
                j               = protocols[k]
                t               = self.t_all[j]
                t_lab           = self.t_label[j]
                EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
                stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
                # for i in range(len(c)):
                #     chan            = c[i]
                stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)& ((stim_table.IPI_ms > 1000*(w+0.02))& (stim_table.Int_cond == Int))|((stim_table.IPI_ms == 0)& (stim_table.Int_prob == Int))]
                stimNum         = stim_list_spef.Number.values
                axs             = fig.add_subplot(gs[0,i])
                resp            = np.nanmean(EEG_block[c, stimNum,:], 0)
                resp_std        = np.nanstd(EEG_block[c, stimNum,:], 0)
                axs.plot(self.x_ax,resp, c=self.color_elab[j], linewidth=3, label= t_lab+', n= '+str(len(stimNum)))
                axs.fill_between(self.x_ax, resp - resp_std,
                                 resp + resp_std,
                                 facecolor=self.color_elab[j], alpha=0.2)
                plt.ylim([-600, 600])
                #plt.title('Intensity: '+str(Int)+'mA', fontsize=10)
                #plt.ylabel('uV', fontsize=10)
                plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=2)
                plt.axvspan(0.02, w, alpha=0.05, color='black')
                if i ==0:
                    plt.yticks([-300, 0, 300])
                else:
                    plt.yticks([-300, 0, 300], ['','',''])
                #plt.title('Intensity: ' + str(Int) + 'mA', fontsize=10)
                #plt.legend()
                plt.legend([],[], frameon=False)#plt.legend()
                plt.xlabel('time [s]', fontsize=10)
                plt.xlim([-lim/2, lim])
                plt.xticks([-lim/2, 0, lim/2, lim], ['','',''])
                plt.tick_params(axis='both', labelsize=10)
                plt.ylabel('', fontsize=10)
                plt.xlabel('', fontsize=10)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/figures/Resp_' + StimChan + '-' + str(self.labels_all[c]) + '_Int'+str(Int)+'_'+str(w)+'s.jpg')
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/figures/Resp_' + StimChan + '-' + str(
            self.labels_all[c]) + '_Int' + str(Int) + '_' + str(w) + 's.svg')
        #plt.show()

    def plot_PP(self, sc, c, protocols, Int, IPI):
        w           = 0.1
        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        print(StimChan)
        print(self.labels_all[c])
        fig = plt.figure(figsize=(10, len(c) * 2))
        plt.suptitle(str(self.subj + "--SP , Stim: " + StimChan + ', Int: ' + str(Int)+'mA'),
                     fontsize=14, y=1)
        gs          = fig.add_gridspec(len(c), len(protocols))  # GridSpec(4,1, height_ratios=[1,2,1,2])
        lim         = np.zeros((len(c),1))
        for k in range(len(protocols)):
            j               = protocols[k]
            t               = self.t_all[j]
            t_lab           = self.t_label[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            for i in range(len(c)):
                chan            = c[i]
                stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)& (stim_table.IPI_ms == IPI)& (stim_table.Int_cond == Int)]
                stimNum         = stim_list_spef.Number.values
                axs             = fig.add_subplot(gs[i, k])
                resp            = np.nanmean(EEG_block[chan, stimNum,:], 0)
                resp_std        = np.nanstd(EEG_block[chan, stimNum,:], 0)
                axs.plot(self.x_ax,resp, c=self.color_elab[j, :], label= t_lab+', n= '+str(len(stimNum)))
                axs.fill_between(self.x_ax, resp - resp_std,
                                 resp + resp_std,
                                 facecolor=self.color_elab[j, :], alpha=0.1)
                if k == 0:
                    lim[i] = max(np.max(abs(resp))+0.05*np.max(abs(resp)),300)
                plt.xlabel('')
                plt.xlim([-0.5, 1])
                plt.ylim([-lim[i], lim[i]])
                #plt.title('Intensity: '+str(Int)+'mA', fontsize=10)
                plt.ylabel(self.labels_all[chan], fontsize=10)
                plt.tick_params(axis='y', labelsize=12)
                plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
                plt.axvline(IPI/1000, ls='--', c=[0, 0, 0], linewidth=1.5)
                plt.xticks([])
                plt.yticks([-300, 0, 300])
                plt.legend()
            plt.xlabel('time [s]')
            plt.xlim([-0.5, 1])
            plt.xticks([-0.5, 0, 0.5, 1])
            plt.tick_params(axis='both', labelsize=12)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/figures/Resp_' + StimChan + '-' + str(self.labels_all[c]) + '_Int'+str(Int)+'_IPI'+str(IPI)+'.jpg')

    def plot_Int_SP(self, sc, c, protocols):
        w = 0.1
        StimChan = self.StimChans[sc]
        ChanP = self.StimChanNums[sc]
        print(StimChan)
        print(self.labels_all[c])
        fig = plt.figure(figsize=(10, len(self.Int_all) * 3))
        plt.suptitle(str(self.subj + "--SP - LL [0.1s], Stim: " + StimChan + ', Resp: ' + self.labels_all[c]),
                     fontsize=14, y=1)
        gs = fig.add_gridspec(len(self.Int_all), 1)  # GridSpec(4,1, height_ratios=[1,2,1,2])
        for j in protocols:
            t               = self.t_all[j]
            t_lab           = self.t_label[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            for i in range(len(self.Int_all)):
                Int             = self.Int_all[i]
                stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)& ((stim_table.IPI_ms > 1000*w+0.02)& (stim_table.Int_cond == Int))|((stim_table.IPI_ms == 0)& (stim_table.Int_prob == Int))]
                stimNum         = stim_list_spef.Number.values
                axs             = fig.add_subplot(gs[i, 0])
                resp            = np.nanmean(EEG_block[c, stimNum,:], 0)
                resp_std        = np.nanstd(EEG_block[c, stimNum,:], 0)
                axs.plot(self.x_ax,resp, c=self.color_elab[j, :], label= t_lab+', n= '+str(len(stimNum)))
                axs.fill_between(self.x_ax, resp - resp_std,
                                 resp + resp_std,
                                 facecolor=self.color_elab[j, :], alpha=0.1)
                plt.xlabel('')
                plt.xlim([-0.5, 1])
                plt.ylim([-800, 800])
                plt.title('Intensity: '+str(Int)+'mA', fontsize=10)
                plt.ylabel('LL SP norm', fontsize=10)
                plt.tick_params(axis='y', labelsize=12)
                plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
                plt.xticks([])
                plt.yticks([-500, 0, 500])
                plt.legend()
        plt.xlabel('time [s]')
        plt.xlim([-0.5, 1])
        plt.xticks([-0.5, 0, 0.5, 1])
        plt.tick_params(axis='both', labelsize=12)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/Resp_1-2-4_SP_' + StimChan + '-' + str(self.labels_all[c]) + '.jpg')
        #plt.show()

    def plot_SP_chron(self, sc, c, protocols, Int):
        w           = 0.1
        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        print(StimChan)
        print(self.labels_all[c])
        fig         = plt.figure(figsize=(20, 8))
        plt.suptitle(str(self.subj + "--SP, Stim: " + StimChan + ', Resp: ' + self.labels_all[c]+', Int: '+str(Int)+'mA'),
                     fontsize=14, y=1)
        gs          = fig.add_gridspec(2, 1)  # GridSpec(4,1, height_ratios=[1,2,1,2])
        LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_CR_' + StimChan + '.csv')
        k           = 0
        win         = 150
        for j in protocols:
            t               = self.t_all[j]
            t_label           = self.t_label[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)& ((stim_table.IPI_ms > win)& (stim_table.Int_cond == Int))|((stim_table.IPI_ms == 0)& (stim_table.Int_prob == Int))]
            stimNum         = stim_list_spef.Number.values
            IPISs           = stim_list_spef.IPI_ms.values
            if j==0:
                mx = len(stimNum)
            resps           = np.zeros((np.int((win/1000+0.01)*self.Fs),)) # 600ms response
            l               = resps.shape[0]
            labels_ttl      = np.zeros((len(stimNum),))
            axs = fig.add_subplot(gs[k, 0])
            for i in range(len(stimNum)):
                resps           = np.concatenate([resps, EEG_block[c, stimNum[i],np.int(4.99*self.Fs):np.int((5+win/1000)*self.Fs)]])
                labels_ttl[i]   = i*l+0.01*self.Fs
                plt.text(i*l + 0.017*self.Fs, 400, str(np.round(LL_all[(LL_all.Chan==c)&(LL_all.IPI==IPISs[i])&(LL_all.Condition==t_label)&(LL_all.Int==Int)]['LL SP'].values[0],1))+'ms/uV', fontsize=8)
                plt.axvline(labels_ttl[i], c=[1, 0, 0], linewidth=2)
            resps = resps[l:]
            plt.plot(resps, color=self.color_elab[j,:])
            plt.xticks(labels_ttl, stim_list_spef.Number.values)  # , rotation='vertical'
            plt.ylabel('[uV]', fontsize=16)
            plt.ylim([-650, 650])
            plt.yticks([-500, 0, 500])
            plt.tick_params(axis='y', labelsize=12)
            plt.xlim(-50, mx*l+50)
            k=1

        plt.savefig(self.path_patient + '/Analysis/Pharmacology/figures/Chron_SP_' + StimChan + '-' + str(self.labels_all[c]) + '_Int' + str(Int) + '.jpg')
    def plot_IO(self, sc, c, protocols):
        # plot all mean, intensity by color gradient
        StimChan = self.StimChans[sc]
        # ChanP           = self.StimChanNums[sc]
        LL_all = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_IO_' + StimChan + '.csv')
        fig = plt.figure(figsize=(10, 8))
        plt.suptitle('IO - Stim: ' + StimChan + ', Resp: ' + self.labels_all[c], fontsize=16)
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        g = sns.pointplot(x="Int", y="LL", hue="Condition", data=LL_all[LL_all.Chan == c], kind="point", ci="sd",
                      height=6,  # make the plot 5 units high
                      aspect=1.5, s=0.8, legend_out=False,
                      palette=sns.color_palette([self.color_elab[0, :], self.color_elab[2, :]]))
        for bar in g.patches:
            bar.set_zorder(3)
        plt.xlabel('Int [mA]', fontsize=16)
        plt.ylabel('LL [uV/ms]', fontsize=16)
        plt.tick_params(axis="both", labelsize=16)
        plt.legend(fontsize=16)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IO/IO_prots_' + StimChan + '-' + str(
            self.labels_all[c]) + '.jpg')
        # plt.show()
    def plot_Int_PP_cat(self, sc, IPI, cat, protocols):
        #cat = 'Entorhinal'
        badchans        = np.concatenate([self.badchans, [self.stim_ind[sc]],[self.stim_ind[sc]-1],[self.stim_ind[sc]+1],[self.stim_ind[sc]+2],[self.stim_ind[sc]-2]])
        c               = np.array(np.where(self.cat_all == cat))
        c               = np.reshape(c, (-1))
        indx            = np.ravel([np.where(c == i) for i in badchans])
        if len(indx)>0:
            indx            = np.array(np.ravel(indx[np.where(indx >= 0)]))
            indx            = np.array([i for i in indx])[:, 0]
            c               = np.delete(c, indx)#np.delete(c, badchans)
        # c = c[c !=badchans]
        if len(c) > 0:
            w               = 0.1
            StimChan        = self.StimChans[sc]
            ChanP           = self.StimChanNums[sc]
            print(StimChan)
            print(self.labels_all[c])
            fig = plt.figure(figsize=(8, len(self.Int_all) * 2))
            plt.suptitle(str(self.subj + "-- PP-" +str(IPI)+"ms, Stim: " + StimChan + ', Resp: ' + str(cat)),
                         fontsize=14, y=1)
            gs = fig.add_gridspec(len(self.Int_all), 1)  # GridSpec(4,1, height_ratios=[1,2,1,2])
            for j in protocols:
                t               = self.t_all[j]
                t_lab           = self.t_label[j]
                EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
                stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
                for i in range(len(self.Int_all)):
                    Int             = self.Int_all[i]
                    stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0) & (stim_table.IPI_ms == IPI ) & (stim_table.Int_cond == Int)]
                    stimNum         = stim_list_spef.Number.values
                    axs             = fig.add_subplot(gs[i, 0])
                    # resp_c          = np.nanmean(EEG_block[c, :,:], axis=0)
                    # resp            = np.nanmean(resp_c[stimNum,:], axis=0)
                    # resp_std        = np.nanstd(resp_c[stimNum,:], axis=0)
                    resp            = np.nanmean(EEG_block[c,stimNum, :], axis=0)
                    resp_std        = np.nanstd(EEG_block[c,stimNum, :], axis=0)

                    axs.plot(self.x_ax,resp, c=self.color_elab[j, :], label= t_lab+', n= '+str(len(stimNum))+', n_chan= '+ str(len(c)))
                    axs.fill_between(self.x_ax, resp - resp_std,
                                     resp + resp_std,
                                     facecolor=self.color_elab[j, :], alpha=0.1)
                    plt.xlabel('')
                    plt.xlim([-0.5, 1])
                    plt.ylim([-800, 800])
                    plt.title('Intensity: '+str(Int)+'mA', fontsize=10)
                    plt.ylabel('[uV]]', fontsize=10)
                    plt.tick_params(axis='y', labelsize=12)
                    plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
                    plt.axvline(IPI/1000, ls='--', c=[0, 0, 0], linewidth=1.5)
                    plt.xticks([])
                    plt.yticks([-500, 0, 500])
                    plt.legend()
            plt.xlabel('time [s]')
            plt.xlim([-0.5, 1])
            plt.xticks([-0.5, 0, 0.5, 1])
            plt.tick_params(axis='both', labelsize=12)
            plt.savefig(self.path_patient + '/Analysis/Pharmacology/figures/Resp_1-2-4_PP_' + StimChan + '-' + str(cat) + '_IPI' + str(np.round(IPI))+'.jpg')
            #plt.show()
    def plot_Int_PP_c(self, sc, IPI, c, protocols):
        #cat = 'Entorhinal'
        w               = 0.1
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        print(StimChan)
        print(self.labels_all[c])
        fig = plt.figure(figsize=(8, len(self.Int_all) * 2))
        plt.suptitle(str(self.subj + "-- PP-" +str(IPI)+"ms, Stim: " + StimChan + ', Resp: ' + str(self.labels_all[c])),
                     fontsize=14, y=1)
        gs = fig.add_gridspec(len(self.Int_all), 1)  # GridSpec(4,1, height_ratios=[1,2,1,2])
        for j in protocols:
            t               = self.t_all[j]
            t_lab           = self.t_label[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            for i in range(len(self.Int_all)):
                Int             = self.Int_all[i]
                stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0) & (stim_table.IPI_ms == IPI ) & (stim_table.Int_cond == Int)]
                stimNum         = stim_list_spef.Number.values
                axs             = fig.add_subplot(gs[i, 0])#fig.add_gridspec(len(self.Int_all), 1)
                # resp_c          = np.nanmean(EEG_block[c, :,:], axis=0)
                # resp            = np.nanmean(resp_c[stimNum,:], axis=0)
                # resp_std        = np.nanstd(resp_c[stimNum,:], axis=0)
                resp            = np.nanmean(EEG_block[c,stimNum, :], axis=0)
                resp_std        = np.nanstd(EEG_block[c,stimNum, :], axis=0)

                axs.plot(self.x_ax,resp, c=self.color_elab[j, :], label= t_lab+', n= '+str(len(stimNum)))
                axs.fill_between(self.x_ax, resp - resp_std,
                                 resp + resp_std,
                                 facecolor=self.color_elab[j, :], alpha=0.1)
                plt.xlabel('')
                plt.xlim([-0.5, 1])
                plt.ylim([-700, 700])
                plt.title('Intensity: '+str(Int)+'mA', fontsize=10)
                plt.ylabel('[uV]]', fontsize=10)
                plt.tick_params(axis='y', labelsize=12)
                plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
                plt.axvline(IPI/1000, ls='--', c=[0, 0, 0], linewidth=1.5)
                plt.xticks([])
                plt.yticks([-500, 0, 500])
                plt.legend()
        plt.xlabel('time [s]')
        plt.xlim([-0.5, 1])
        plt.xticks([-0.5, 0, 0.5, 1])
        plt.tick_params(axis='both', labelsize=12)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/figures/Resp_1-2-4_PP_' + StimChan + '-' + str(self.labels_all[c]) + '_IPI' + str(np.round(IPI))+'.jpg')
        #plt.show()

    def plot_PP_ex(self, sc,  c, IPIs, Ints, protocols):
        w=0.25
        # example of chosen IPIs (column) and protocols (overlap) for each conditioning intensity (rows)
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        print(StimChan)
        print(self.labels_all[c])
        fig = plt.figure(figsize=(len(IPIs)*9, len(Ints)*3+2))
        plt.suptitle(str(self.subj + "-- PP,  Stim: " + StimChan + ', Resp: ' + str(self.labels_all[c])),
                     fontsize=10, y=1)
        gs = fig.add_gridspec(len(Ints), len(IPIs))  # GridSpec(4,1, height_ratios=[1,2,1,2])
        for j in protocols:
            t               = self.t_all[j]
            t_lab           = self.t_label[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            for k in range(len(IPIs)):
                for i in range(len(Ints)):
                    IPI             = IPIs[k]
                    Int             = Ints[i]
                    stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0) & (stim_table.IPI_ms == IPI ) & (stim_table.Int_cond == Int)]
                    stimNum         = stim_list_spef.Number.values
                    axs             = fig.add_subplot(gs[i, k])#fig.add_gridspec(len(self.Int_all), 1)
                    resp            = np.nanmean(EEG_block[c,stimNum, :], axis=0)
                    resp_std        = np.nanstd(EEG_block[c,stimNum, :], axis=0)

                    axs.plot(self.x_ax,resp, c=self.color_elab[j], label= t_lab+', n= '+str(len(stimNum)))
                    axs.fill_between(self.x_ax, resp - resp_std,
                                     resp + resp_std,
                                     facecolor=self.color_elab[j], alpha=0.1)
                    plt.xlabel('')
                    plt.yticks([])
                    plt.xlim([-0.5, 2])
                    plt.ylim([-600, 600])
                    if i==0: plt.title('IPI: '+str(IPI)+'ms', fontsize=10)
                    if k == 0: plt.ylabel(str(Int)+'mA', fontsize=10)
                    plt.yticks([-300, 0, 300])
                    plt.tick_params(axis='y', labelsize=12)
                    plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
                    plt.axvline(IPI/1000, ls='--', c=[0, 0, 0], linewidth=1.5)
                    plt.xticks([])
                    plt.axvspan(0.02+IPI/1000, w+IPI/1000, alpha=0.05, color='black')
                    #plt.legend()
                plt.xlabel('time [s]')
                plt.xlim([-0.1, np.max(IPIs)/1000+0.3])
                plt.xticks([0, 0.5,1, 1.5, 2])
                plt.tick_params(axis='both', labelsize=12)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/figures/PP_Int1_' + StimChan + '-' + str(self.labels_all[c]) + '_IPI.jpg')
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/figures/PP_Int1_' + StimChan + '-' + str(
            self.labels_all[c]) + '_IPI.svg')
        #plt.show()

    def plot_LL_BL_IPI_c(self, sc, c):
        #### IPI plot
        #### LL  normalized by basline 2mA response
        ### chosen one for paper
        w           = 0.1
        StimChan    = self.StimChans[sc]
        print(StimChan)
        print(self.labels_all[c])
        LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan + '.csv')

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        #plt.suptitle(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + str(self.labels_all[c])), fontsize=20)
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        data        = LL_all[(LL_all['Int'] > 0) & (LL_all['Chan'] == c)]
        mx          = np.max(data['LL PP norm BL'])
        grid        = sns.catplot(x="IPI", y='LL PP norm BL', hue="Condition", data=data, row='Int', kind="swarm",
                    height=4,  # make the plot 5 units high
                    aspect=2.2, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
        plt.ylim([0, mx+0.3])
        #plt.xticks([0, 2, 5, 7, 9,11, 13, 15], ['100', '150', '250', '350', '500', '700', '1000', '1600'])
        plt.xticks([0,6,13,18,22,26,28], ['10', '30', '100', '250', '500', '1000', '1600'])
        plt.xlabel('IPI [ms]', fontsize=18)
        plt.ylabel('LL norm BL', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)

        grid.axes[0, 0].set_title(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + str(self.labels_all[c])), fontsize=20)
        grid.axes[0, 0].set_ylabel('LL norm BL', fontsize=18)
        grid.axes[0, 0].tick_params(axis='y', labelsize=14)
        grid.axes[0, 0].axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
        grid.axes[1, 0].set_ylabel('LL norm BL', fontsize=18)
        grid.axes[1, 0].tick_params(axis='y', labelsize=14)
        grid.axes[1, 0].axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_normBL_IPI_'+StimChan+ '-'+str(self.labels_all[c])+'.jpg')
        #plt.show()
    def plot_LL_BL_IPI_cat(self, sc, cat,w, IO=False):
        #### IPI plot
        #### LL  normalized by basline 2mA response
        ### chosen one for paper
        # find channels
        cs          = np.array(np.where(self.cat_all == cat))  # find indeces of specific cat
        cs          = np.reshape(cs, (-1))
        StimChan    = self.StimChans[sc]

        try: # load LL, calculate LL if non existing
            LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan +'_'+str(w)+'s.csv')
        except IOError:
            self.get_LL_sc(sc=sc, protocols=[0, 2], w=w)
            LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan +'_'+str(w)+'s.csv')
        LL_IO   = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_IO_' + StimChan + '_' + str(w) + 's.csv')
        LL_all  = LL_all[LL_all['Chan'].isin(cs) & (LL_all['Int'] > 0)]
        LL_all.insert(3, "LL normed", 0, True)
        ###both protocols
        data_PP = LL_all[(LL_all['IPI'] > 1000 * w) & (LL_all['Chan'].isin(cs))]
        # num             = len(LL_all[(LL_all['IPI'] > 1000*w) &(LL_all['Int'] ==Int) & (LL_all['Chan']==cs[0])]['LL'].values)
        data_SP = LL_IO[LL_IO['Chan'].isin(cs)]
        data_SP.insert(3, "LL SP", data_SP['LL'], True)
        if IO:
            data_PP = pd.concat([data_SP, data_PP], sort=False)
        for c in cs:
            ref = np.mean(data_PP[(data_PP['Chan'] == c) & (data_PP['Condition'] == 'Baseline') & (data_PP['Int'] == 2)][
                    'LL SP'])
            LL_all.loc[(LL_all['Chan'] == c), 'LL normed'] = LL_all[(LL_all['Chan'] == c)]['LL PP'].values / ref

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")

        #sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        sns.set_palette(['#594157', "#8FB996"])
        #my_color_palette = {"Dz 5mg/kg": "#8FB996", "NaCl": '#594157', "PTZ 20mg/kg": "#F1BF98"}
        #data        = LL_all[LL_all['Chan'].isin(cs)&(LL_all['Int'] > 0)]
        mx          = np.max(LL_all['LL normed'])
        grid        = sns.catplot(x="IPI", y='LL normed', hue="Condition", data=LL_all, row='Int', kind="swarm",
                    height=4,  # make the plot 5 units high
                    aspect=3.3, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
        plt.ylim([0, mx+0.3])
        plt.xticks([0,6,13,18,22,26,28], ['10', '30', '100', '250', '500', '1000', '1600'])
        plt.xlabel('IPI [ms]', fontsize=14)
        plt.ylabel('LL norm BL', fontsize=14)
        plt.tick_params(axis='both', labelsize=12)
        plt.axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)

        #grid.axes[0, 0].set_title(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + cat+', LL: '+str(w)+'s'), fontsize=14)
        grid.axes[0, 0].set_ylabel('LL norm BL', fontsize=14)
        grid.axes[0, 0].tick_params(axis='y', labelsize=12)
        grid.axes[0, 0].axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
        grid.axes[1, 0].set_ylabel('LL norm BL', fontsize=14)
        grid.axes[1, 0].tick_params(axis='y', labelsize=12)
        grid.axes[1, 0].axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.suptitle(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + cat + ', LL: ' + str(w) + 's'), fontsize=14, y=1.01)

        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_IPI_'+StimChan+ '-'+cat[0:3]+'_'+str(w)+'.jpg')
        plt.savefig(
            self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_IPI_' + StimChan + '-' + cat[0:3] + '_' + str(
                w) + '.svg')

        #plt.show()
    def plot_LL_dist_IPI_c(self, sc, c):
        w           = 0.1
        StimChan    = self.StimChans[sc]
        print(StimChan)
        print(self.labels_all[c])
        LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_CR_' + StimChan + '.csv')

        LL_all.insert(7, "Dist", 0, True)

        LL_all.loc[(LL_all['Chan'] == c), 'Dist'] = (LL_all[(LL_all['Chan'] == c) ]['LL PP norm BL'].values - LL_all[(LL_all['Chan'] == c) ]['LL SP norm BL'].values)* sin(45)
        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        data        = LL_all[(LL_all['IPI'] > 100) & (LL_all['Int'] > 0) & (LL_all['Chan'] == c)]
        mx          = np.max(data['Dist'])
        mn          = np.min(data['Dist'])
        grid        = sns.catplot(x="IPI", y='Dist', hue="Condition", data=data, row='Int', kind="swarm",
                    height=3.5,  # make the plot 5 units high
                    aspect=1.8, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
        plt.ylim([mn-0.3, mx+0.2])
        plt.xticks([0, 2, 5, 7, 9,11, 13, 15], ['100', '150', '250', '350', '500', '700', '1000', '1600'])
        plt.xlabel('IPI [ms]', fontsize=18)
        plt.ylabel('dist do diagonal', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.axhline(0, ls='--', c=[0,0,0], linewidth=1.5)

        grid.axes[0, 0].set_title(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + str(self.labels_all[c])), fontsize=20)
        grid.axes[0, 0].set_ylabel('dist do diagonal', fontsize=18)
        grid.axes[0, 0].tick_params(axis='y', labelsize=14)
        grid.axes[0, 0].axhline(0, ls='--', c=[0,0,0], linewidth=1.5)
        grid.axes[1, 0].set_ylabel('dist do diagonal', fontsize=18)
        grid.axes[1, 0].tick_params(axis='y', labelsize=14)
        grid.axes[1, 0].axhline(0, ls='--', c=[0,0,0], linewidth=1.5)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_dist_IPI_'+StimChan+ '-'+str(self.labels_all[c])+'.jpg')

    def plot_LL_cond_IPI_c(self, sc, c):
        w           = 0.1
        StimChan    = self.StimChans[sc]
        print(StimChan)
        print(self.labels_all[c])
        LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_CR_' + StimChan + '.csv')

        LL_all.loc[(LL_all['Chan'] == c) & (LL_all['Condition'] == 'Benzodiazepin'), 'LL PP norm Cond'] = LL_all[(LL_all['Chan'] == c) & (LL_all['Condition'] == 'Benzodiazepin')]['LL PP norm Cond'].values - (
                        1 - (self.SP_LL_mean[c, 2] / self.SP_LL_mean[c, 0]))

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        data        = LL_all[(LL_all['IPI'] > 100) & (LL_all['Int'] > 0) & (LL_all['Chan'] == c)]
        mx          = np.max(data['LL PP norm Cond'])
        grid        = sns.catplot(x="IPI", y='LL PP norm Cond', hue="Condition", data=data, row='Int', kind="swarm",
                    height=3.5,  # make the plot 5 units high
                    aspect=1.8, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
        plt.ylim([0, mx+0.3])
        plt.xticks([0, 2, 5, 7, 9,11, 13, 15], ['100', '150', '250', '350', '500', '700', '1000', '1600'])
        plt.xlabel('IPI [ms]', fontsize=18)
        plt.ylabel('LL norm Cond', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.axhline(1, ls='--', c=self.color_elab[0, :], linewidth=1.5)
        plt.axhline(self.SP_LL_mean[c, 2] / self.SP_LL_mean[c, 0], ls='--', c=self.color_elab[2, :],
                    linewidth=1.5)

        grid.axes[0, 0].set_title(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + str(self.labels_all[c])), fontsize=20)
        grid.axes[0, 0].set_ylabel('LL norm Cond', fontsize=18)
        grid.axes[0, 0].tick_params(axis='y', labelsize=14)
        grid.axes[0, 0].axhline(1, ls='--', c=self.color_elab[0, :], linewidth=1.5)
        grid.axes[0, 0].axhline(self.SP_LL_mean[c, 2] / self.SP_LL_mean[c, 0], ls='--', c=self.color_elab[2, :],
                    linewidth=1.5)
        grid.axes[1, 0].set_ylabel('LL norm Cond', fontsize=18)
        grid.axes[1, 0].tick_params(axis='y', labelsize=14)
        grid.axes[1, 0].axhline(1, ls='--', c=self.color_elab[0, :], linewidth=1.5)
        grid.axes[1, 0].axhline(self.SP_LL_mean[c, 2] / self.SP_LL_mean[c, 0], ls='--', c=self.color_elab[2, :],
                    linewidth=1.5)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_normCond_IPI_'+StimChan+ '-'+str(self.labels_all[c])+'.jpg')
        #plt.show()

    def plot_LL_ratio_IPI_c(self, sc, c):
        w           = 0.1
        StimChan    = self.StimChans[sc]
        print(StimChan)
        print(self.labels_all[c])
        LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_CR_' + StimChan + '.csv')

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])

        data = LL_all[(LL_all['IPI'] > 100) & (LL_all['Int'] > 0) & (LL_all['Chan'] == c)]
        mx = np.max(data['LL PP/SP'])
        grid = sns.catplot(x="IPI", y='LL PP/SP', hue="Condition", data=data, row='Int', kind="swarm",
                    height=3.5,  # make the plot 5 units high
                    aspect=1.8, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
        plt.ylim([0, mx+0.3])
        plt.xticks([0, 2, 5, 7, 9,11, 13, 15], ['100', '150', '250', '350', '500', '700', '1000', '1600'])
        plt.xlabel('IPI [ms]', fontsize=18)
        plt.ylabel('LL Ratio', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)

        grid.axes[0, 0].set_title(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + str(self.labels_all[c])), fontsize=20)
        grid.axes[0, 0].set_ylabel('LL Ratio', fontsize=18)
        grid.axes[0, 0].tick_params(axis='y', labelsize=14)
        grid.axes[0, 0].axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
        grid.axes[1, 0].set_ylabel('LL Ratio', fontsize=18)
        grid.axes[1, 0].tick_params(axis='y', labelsize=14)
        grid.axes[1, 0].axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_Ratio_IPI_'+StimChan+ '-'+str(self.labels_all[c])+'.jpg')
        #plt.show()

    def plot_LL_abs_IPI_c(self, sc, c):
        w           = 0.1
        StimChan    = self.StimChans[sc]
        print(StimChan)
        print(self.labels_all[c])
        LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_CR_' + StimChan + '.csv')

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        sns.set_palette([self.color_elab[0, :], self.color_elab[2, :]])
        data = LL_all[(LL_all['IPI'] > 100) & (LL_all['Int'] > 0) & (LL_all['Chan'] == c)]
        mn      = np.min(data['LL PP'])
        mx      = np.max(data['LL PP'])
        grid = sns.catplot(x="IPI", y='LL PP', hue="Condition", data=data, row='Int', kind="swarm",
                    height=3.5,  # make the plot 5 units high
                    aspect=1.8, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
        plt.ylim([mn-5, mx+3])
        plt.xticks([0, 2, 5, 7, 9,11, 13, 15], ['100', '150', '250', '350', '500', '700', '1000', '1600'])
        plt.xlabel('IPI [ms]', fontsize=18)
        plt.ylabel('LL [uv/ms]', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)

        grid.axes[0, 0].set_title(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + str(self.labels_all[c])), fontsize=20)
        grid.axes[0, 0].set_ylabel('LL [uv/ms]', fontsize=18)
        grid.axes[0, 0].tick_params(axis='y', labelsize=14)
        grid.axes[1, 0].set_ylabel('LL [uv/ms]', fontsize=18)
        grid.axes[1, 0].tick_params(axis='y', labelsize=14)
        plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_abs_IPI_'+StimChan+ '-'+str(self.labels_all[c])+'.jpg')
        #plt.show()

    def plot_LL_scatter_cat(self, sc, cat, Int=2, w=0.1):
        # cat = 'Entorhinal'
        badchans    = np.concatenate([self.badchans, [self.stim_ind[sc]], [self.stim_ind[sc] - 1], [self.stim_ind[sc] + 1],
                                   [self.stim_ind[sc] + 2], [self.stim_ind[sc] - 2]])
        c           = np.array(np.where(self.cat_all == cat))
        c           = np.reshape(c, (-1))
        indx        = np.ravel([np.where(c == i) for i in badchans])
        if len(indx) > 0:
            indx    = np.array(np.ravel(indx[np.where(indx >= 0)]))
            indx    = np.array([i for i in indx])[:, 0]
            c       = np.delete(c, indx)  # np.delete(c, badchans)
        if len(c) > 0:
            StimChan    = self.StimChans[sc]
            print(StimChan)
            LL_all          = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan + '.csv')
            num_IPI         = len(np.unique(np.sort(LL_all[(LL_all['IPI']>w*1000)&(LL_all['Int']==Int)]['IPI'])))
            colors_IPI      = np.zeros((num_IPI, 3))
            colors_IPI[:, 0] = np.linspace(0, 1, num_IPI)
            ## norm LL over IPI
            plt.figure(figsize=(10, 10))
            plt.title(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + str(cat)),
                         fontsize=14)
            plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
            # insert gap
            dataBL      = LL_all[(LL_all['IPI'] > w * 1000) & (LL_all['Int'] == Int) & (LL_all['Chan'].isin(c))& (LL_all['Condition']=='Baseline')]
            dataBZD     = LL_all[(LL_all['IPI'] > w * 1000) & (LL_all['Int'] == Int) & (LL_all['Chan'].isin(c)) & (
                        LL_all['Condition'] == 'Benzodiazepin')]
            dBL         = np.nanmean((dataBL['LL PP norm BL'] - dataBL['LL SP norm BL'])* sin(45))
            dBZD        = np.nanmean((dataBZD['LL PP norm BL'] - dataBZD['LL SP norm BL']) * sin(45))


            #data.plot.scatter(x="LL SP norm BL", y="LL PP norm BL", c='IPI', colormap='hot')
            sns.scatterplot(data=LL_all[(LL_all['IPI']>w*1000)&(LL_all['Int']==Int)&(LL_all['Chan'].isin(c))], x="LL SP norm BL", y="LL PP norm BL", hue="IPI", style="Condition", s=100, palette=sns.color_palette(colors_IPI))
            plt.xlabel('Cond. Pulse norm', fontsize=18)
            plt.ylabel('Prob. Pulse norm', fontsize=18)
            plt.tick_params(axis='both', labelsize=15)
            plt.xticks(np.arange(0, 3.1, 0.5))
            plt.yticks(np.arange(0, 3.1, 0.5))
            plt.ylim([0, 3])
            plt.xlim([0.5, 1.5])
            plt.plot([0, 4], [0, 4], ls='--', c=[0, 0, 0], linewidth=1)
            plt.axvline(1, ls='--', c=[0, 0, 0], linewidth=1)
            plt.axhline(1, ls='--', c=[0, 0, 0], linewidth=1)
            plt.text(1.3,2.7, 'Mean Dist to Diagonal - BL: '+str(np.round(dBL,2)))
            plt.text(1.3, 2.5, 'Mean Dist to Diagonal - BZD: ' + str(np.round(dBZD, 2)))

            plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_norm_SS_'+StimChan+ '-'+str(cat)+'.jpg')
            #plt.show()
        else:
            print('no responses')
    def plot_LLnorm_IPI_cat(self, sc, cat):
        # cat = 'Entorhinal'
        badchans = np.concatenate([self.badchans, [self.stim_ind[sc]], [self.stim_ind[sc] - 1], [self.stim_ind[sc] + 1],
                                   [self.stim_ind[sc] + 2], [self.stim_ind[sc] - 2]])
        c = np.array(np.where(self.cat_all == cat))
        c = np.reshape(c, (-1))
        indx = np.ravel([np.where(c == i) for i in badchans])
        if len(indx) > 0:
            indx = np.array(np.ravel(indx[np.where(indx >= 0)]))
            indx = np.array([i for i in indx])[:, 0]
            c = np.delete(c, indx)  # np.delete(c, badchans)
        if len(c) > 0:
            w           = 0.1
            StimChan    = self.StimChans[sc]
            print(StimChan)
            print(self.labels_all[c])
            LL_all      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_CR_' + StimChan + '.csv')

            # insert gap to demonstrtate difference in mean
            for i  in c:
                LL_all.loc[(LL_all['Chan'] == i) & (LL_all['Condition'] == 'Benzodiazepin'), 'LL PP norm Cond'] = \
                LL_all[(LL_all['Chan'] == i) & (LL_all['Condition'] == 'Benzodiazepin')]['LL PP norm Cond'].values -(1-(self.SP_LL_mean[i, 2]/self.SP_LL_mean[i, 0]))
            #data        = LL_all[(LL_all['IPI'] > w * 1000) & (LL_all['Int'] == Int) & (LL_all['Chan'].isin(c))]

            ## norm LL over IPI
            plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
            sns.set_palette([self.color_elab[0, :],self.color_elab[2, :]])
            grid = sns.catplot(x="IPI", y="LL PP norm Cond", hue="Condition", data=LL_all[(LL_all['Int'] > 0)&(LL_all['Chan'].isin(c))], row='Int', kind="swarm",
                        height=3.5,  # make the plot 5 units high
                        aspect=2.5, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
            plt.ylim([0, 3])
            plt.xticks([0, 6, 13, 18, 22, 26, 28], ['10', '30', '100', '250', '500', '1000', '1600'])
            plt.xlabel('IPI [ms]', fontsize=18)
            plt.ylabel('LL norm', fontsize=18)
            plt.tick_params(axis='both', labelsize=18)
            plt.axhline(1, ls='--', c=self.color_elab[0, :], linewidth=1.5)
            plt.axhline(self.SP_LL_mean[i, 2] / self.SP_LL_mean[i, 0], ls='--', c=self.color_elab[2, :],
                                    linewidth=1.5)
            plt.text(1, 2.5, 'SP mean Ratio: '+str(np.round((self.SP_LL_mean[i, 2]/self.SP_LL_mean[i, 0]),2)), fontsize=12)

            grid.axes[0, 0].set_title(str(self.subj + "-- Stim: " + StimChan + ', Resp: ' + str(cat)), fontsize=20)
            grid.axes[0, 0].set_ylabel('LL norm', fontsize=18)
            grid.axes[0, 0].tick_params(axis='y', labelsize=18)
            grid.axes[0, 0].axhline(1, ls='--', c=self.color_elab[0, :], linewidth=1.5)
            grid.axes[0, 0].axhline(self.SP_LL_mean[i, 2] / self.SP_LL_mean[i, 0], ls='--', c=self.color_elab[2, :],
                                    linewidth=1.5)
            grid.axes[1, 0].set_ylabel('LL norm', fontsize=18)
            grid.axes[1, 0].tick_params(axis='y', labelsize=18)
            grid.axes[1, 0].axhline(1, ls='--', c=self.color_elab[0,:], linewidth=1.5)
            grid.axes[1, 0].axhline(self.SP_LL_mean[i, 2]/self.SP_LL_mean[i, 0], ls='--', c=self.color_elab[2,:], linewidth=1.5)
            plt.savefig(self.path_patient + '/Analysis/Pharmacology/LL/figures/IPI/LL_norm_IPI_'+StimChan+ '-'+str(cat)+'.jpg')
            #plt.show()
        else:
            print('no responses')

    def plot_WT(self,sc, c, IPI, Int, protocols):
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        print(StimChan)
        for j in protocols:
            t               = self.t_all[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)& (stim_table.Int_cond == Int)& (stim_table.IPI_ms == IPI)]
            stimNum         = stim_list_spef.Number.values
            data            = EEG_block[[c], [stimNum], :]

            WT_funcs = plot_WT_funcs.main(data = data, Fs=self.Fs, IPI=IPI, Int=Int)
            plt.figure(figsize=(30, 12))
            WT_funcs.plot_WT_mean(c=c, x_ax=self.x_ax)
            plt.suptitle(str(self.subj + " -- "+self.t_label[j]+", Stim: " + StimChan + ', Resp: ' + self.labels_all[c] + ', Int: ' + str(Int) + 'mA, IPI: '+str(IPI)+'ms, n='+str(len(stimNum))),
                         fontsize=14, y=1)
            plt.savefig(self.path_patient + '/Analysis/Pharmacology/freq_analysis/WT_PP_'+ self.t_label[j]+"-" + StimChan + '-' + self.labels_all[c] + '-Int' + str(Int) + '-IPI '+str(IPI)+'.jpg')

    def plot_WT_IO(self,sc, c, Int, protocols):
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        print(StimChan)
        IPI = 0
        for j in protocols:
            t               = self.t_IO[j]# self.t_all[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)& (stim_table.Int_prob == Int)]
            stimNum         = stim_list_spef.Number.values
            data            = EEG_block[[c], [stimNum], :]

            WT_funcs        = plot_WT_funcs.main(data = data, Fs=self.Fs, IPI=IPI, Int=Int)
            plt.figure(figsize=(30, 12))
            WT_funcs.plot_WT_mean(c=c, x_ax=self.x_ax)
            plt.suptitle(str(self.subj + " -- "+self.t_label[j]+", Stim: " + StimChan + ', Resp: ' + self.labels_all[c] + ', Int: ' + str(Int) + 'mA, n='+str(len(stimNum))),
                         fontsize=14, y=1)
            plt.savefig(self.path_patient + '/Analysis/Pharmacology/freq_analysis/WT_IO_'+ self.t_label[j]+"-" + StimChan + '-' + self.labels_all[c] + '-Int' + str(Int) + '.jpg')
    def save_WT(self,sc,c,t_prot):
        #only calculated once. its time consuming
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t_prot + '.npy')
        stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t_prot + '.csv')

        pwr, _, f, phase, _ = freq_funcs.get_WT(data=EEG_block[[c],:,:],Fs=self.Fs)  # (100, 1, 48, 5000), (#f, #c, trials, time in seconds)

        np.savez(self.path_patient + '/Analysis/Pharmacology/freq_analysis/WT/WT_'+StimChan+'_'+self.labels_all[c]+'_'+t_prot+'.npz', pwr=pwr,  f=f,phase=phase)
        return pwr, f
    def plot_freq_zscore(self,sc, c, protocols, band):
        Ints = [0.4, 1,2,4,8]
        StimChan        = self.StimChans[sc]
        ChanP           = self.StimChanNums[sc]
        print(StimChan)
        plt.figure(figsize=(10, len(Ints)*3))
        plt.suptitle(str(self.subj + " -- "+self.f_bands_label[band]+" Zscore,  Stim: " + StimChan + ', Resp: ' + self.labels_all[
            c]),fontsize=14, y=1)
        for j in protocols:
            t               = self.t_IO[j]# self.t_all[j]
            EEG_block       = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table      = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            try:
                WT = np.load(self.path_patient + '/Analysis/Pharmacology/freq_analysis/WT/WT_'+StimChan+'_'+self.labels_all[c]+'_'+t+'.npz')
                pwr = WT['pwr']
                f = WT['f']
            except IOError:
                pwr, f = self.save_WT(sc,c, t)

            stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)]
            stim_list_spef.insert(0, 'ID', np.arange(len(stim_list_spef)), True)
            stimNum         = stim_list_spef.Number.values
            data            = EEG_block[[c], [stimNum], :]
            #pwr, pwr_mean, f, _, _ = freq_funcs.get_WT(data, Fs=self.Fs) #(100, 1, 48, 5000), (#f, #c, trials, time in seconds)
            pwr             = pwr[:, : , stimNum, :] # all freqs, chosen channel, specific trials, all datapoints
            mn              = np.nanmean(pwr[np.where((f >= self.f_bands[band,0]) & (f <= self.f_bands[band,1])),0,:,1500:2000]) #mean of high gamma for each trial and each timepoint
            std             = np.nanstd(pwr[np.where((f >= self.f_bands[band,0]) & (f <= self.f_bands[band,1])),0,:,1500:2000])
            pwr_z           = (np.nanmean(pwr[np.array(np.where((f >= self.f_bands[band,0]) & (f <= self.f_bands[band,1])))[0,:],0,:,:],axis=0) - mn)/std
            for i in range(len(Ints)):
                Int = Ints[i]
                plt.subplot(len(Ints), 2, i*2+1)
                ix = stim_list_spef.loc[stim_list_spef.Int_prob==Int, 'ID'].values
                #zscore
                plt.plot(self.x_ax, np.mean(pwr_z[ix,:], axis=0), c=self.color_elab[j,:])
                plt.fill_between(self.x_ax, np.mean(pwr_z[ix,:], axis=0) - np.std(pwr_z[ix,:], axis=0),
                                 np.mean(pwr_z[ix,:], axis=0) + np.std(pwr_z[ix,:], axis=0), color=self.color_elab[j,:],alpha=0.1)
                plt.xlim(-3, 3)
                plt.ylim(-5, 30)
                plt.ylabel(str(Int)+' mA')
                plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
                if i == len(Ints)-1:
                    plt.xlabel('time [s]')
                #EEG
                plt.subplot(len(Ints), 2, i * 2 + 2)
                plt.plot(self.x_ax, np.mean(data[0,ix, :], axis=0),c=self.color_elab[j,:], label=self.t_label[j]+', n: '+str(len(ix)))
                plt.fill_between(self.x_ax, np.mean(data[0,ix, :], axis=0) - np.std(data[0,ix, :], axis=0),
                                 np.mean(data[0,ix, :], axis=0) + np.std(data[0,ix, :], axis=0), alpha=0.1,color=self.color_elab[j,:])
                plt.xlim(-3, 3)
                plt.ylim(-800, 800)
                plt.ylabel('uV')
                plt.legend()
                plt.axvline(0, ls='--', c=[0, 0, 0], linewidth=1.5)
                if i == len(Ints)-1:
                    plt.xlabel('time [s]')
        plt.savefig(
            self.path_patient + '/Analysis/Pharmacology/freq_analysis/figures/'+self.f_bands_label[band]+'_' + StimChan + '-' +
            self.labels_all[c] + '.jpg')
    def plot_WT_all(self, sc, c,  protocols):
        IPI         = 0
        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        print(StimChan)
        for j in protocols:
            t           = self.t_all[j]
            EEG_block = np.load(self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.npy')
            stim_table = pd.read_csv(
                self.path_patient + '/Analysis/Pharmacology/data/All_resps_Ph_' + t + '.csv')
            stim_list_spef = stim_table[
                (stim_table.ChanP == ChanP) & (stim_table.noise == 0)]
            IPI            = np.array(stim_list_spef.IPI_ms.values)
            stimNum         = stim_list_spef.Number.values
            data            = EEG_block[[c], [stimNum], :]
            t_0 = 5
            win = 3
            # second pulse
            LL_start    = np.round((IPI / 1000 + t_0) * self.Fs)  # start position at sencond trigger plus 20ms (art removal)
            LL_end      = np.round((IPI / 1000 + t_0 + win) * self.Fs) - 1
            inds        = np.linspace(LL_start, LL_end, win*self.Fs).T.astype(int)
            inds        = np.expand_dims(inds, axis=0)

            data        = np.take_along_axis(data, inds, axis=2)
            x_ax        = np.arange(0, 3, (1 / self.Fs))
            WT_funcs    = plot_WT_funcs.main(data=data, Fs=self.Fs, IPI=0, Int=2)
            plt.figure(figsize=(20, 12))
            WT_funcs.plot_WT_resp(c=c, x_ax=x_ax)
            plt.suptitle(str(
                self.subj + " -- " + self.t_label[j] + ", All  Stim: " + StimChan + ', Resp: ' + self.labels_all[
                    c] + ', n=' + str(len(stimNum))),
                         fontsize=14, y=1)
            plt.savefig(self.path_patient + '/Analysis/Pharmacology/freq_analysis/WT_all_' + self.t_label[
                j] + "-" + StimChan + '-' + self.labels_all[c] +  '.jpg')
            #plt.show()
    def plot_LLnorm_IPI_rState(self, sc, c):
        w = 0.1
        StimChan = self.StimChans[sc]
        ChanP = self.StimChanNums[sc]
        print(StimChan)
        print(self.labels_all[c])
        LL_all = pd.read_csv(self.path_patient + '/Analysis/Pharmacology/LL//LL_CR_' + StimChan + '.csv')

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        #plt.suptitle(
        #    str('LL [' + str(w) + 's]--- ' + self.subj + ", Stim: " + StimChan + ', Resp: ' + self.labels_all[c]),
        #    fontsize=20, y=1)
        # sns.set(font_scale=2)
        grid = sns.catplot(x="IPI", y="LL PP norm", hue="Int", data=LL_all[(LL_all['Int'] > 0)&(LL_all['Chan'].isin(c))], row='State_label', kind="swarm",
                    height=3,  # make the plot 5 units high
                    aspect=2.5, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
        plt.ylim([0, 3])
        plt.xticks([0, 6, 13, 18, 22, 26, 28], ['10', '30', '100', '250', '500', '1000', '1600'])
        plt.xlabel('IPI [ms]', fontsize=18)
        plt.ylabel('LL norm', fontsize=18)
        plt.tick_params(axis='both', labelsize=18)
        plt.axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)

        grid.axes[0, 0].set_ylabel('LL norm', fontsize=18)
        grid.axes[0, 0].tick_params(axis='y', labelsize=18)
        grid.axes[0, 0].axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
        #grid.axes[0, 0].set_title(str('LL [' + str(w) + 's norm]--- ' + self.subj + ", Stim: " + StimChan + ', Resp: ' + self.labels_all[c]+' -- Baseline(top) vs Benzo (Bottom)'),fontsize=15, y=1.1)

        grid.axes[1, 0].set_title(' ',fontsize=12)
        # plt.savefig(path_patient + '/Analysis/Circadian_PP/PP/LL_norm_Sleep_'+StimChan+ '-'+str(labels_all[c])+'.jpg')
        plt.show()