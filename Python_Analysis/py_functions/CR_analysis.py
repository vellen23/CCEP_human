import os
import numpy as np
import mne
import h5py
#import scipy.fftpack
import matplotlib
import platform
import pywt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
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
from pandas import read_excel
from scipy.stats import norm

class main:
    def __init__(self,subj, protocol_type):

        if platform.system()=='Windows':
            sep = ','
        else: #'Darwin' for MAC
            sep =';'

        self.color_elab         = np.zeros((3, 3))
        self.color_elab[0, :]   = np.array([31, 78, 121]) / 255
        self.color_elab[1, :]   = np.array([189, 215, 238]) / 255
        self.color_elab[2, :]   = np.array([0.256, 0.574, 0.431])

        self.subj           = subj
        self.pt             = protocol_type
        cwd                 = os.getcwd()
        self.path_patient   = os.path.dirname(os.path.dirname(cwd)) + '/Patients/' + subj
        labels_all          = pd.read_csv(self.path_patient + "/infos/" + subj + "_BP_labels.csv", header=0, dtype=str, sep=sep)
        self.labels_all     = labels_all.label.values
        data                = pd.read_csv(self.path_patient + "/infos/" + subj + "_BP_labels.csv", header=0, sep=sep)
        file_name           = subj + '_lookup.xlsx'
        df                  = pd.read_excel(os.path.join(self.path_patient + "/infos/", file_name),sheet_name='Par_circ')  # ChanP  ChanN  Int [mA]  IPI [ms]  ISI [s]  Num_P
        stim_chan = np.array(df.values[:, 0:2], dtype=float)
        stim_chan = stim_chan[~np.isnan(stim_chan)]
        stim_chan = stim_chan[np.nonzero(stim_chan)].reshape(-1, 2)
        self.StimChanNums = stim_chan[:, 0]

        self.StimChans = []  # np.zeros((len(stim_chan)))
        for i in range(len(stim_chan)):
            self.StimChans.append(self.labels_all[(np.array(data.chan_BP_P.values) == stim_chan[i, 0]) & (np.array(data.chan_BP_N.values) == stim_chan[i, 1])][0])

        self.IPI_all = df.IPI.values
        Int_all = df.Int_cond.values
        self.Int_all = Int_all[~np.isnan(Int_all)]
        Int_p = df.Int_prob.values
        self.Int_p = Int_p[~np.isnan(Int_p)]

        self.Fs = 500 #todo: remove hardcode
        self.dur = np.zeros((1, 2), dtype=np.int)
        self.dur[0, :] = [-5, 5]
        self.dur_tot = np.int(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

    def cut_blocks_stim(self, exp, block_num):
        #bad_chans = []
        for u in range(len(self.StimChans)):
            StimChan        = self.StimChans[u]
            ChanP           = self.StimChanNums[u]
            EEG_stims = np.zeros((len(self.labels_all), 1, self.dur_tot * self.Fs))
            print(StimChan)
            badchans       =  pd.read_excel(self.path_patient + '/Data/experiment' + str(exp) + '/' + self.subj + "_stimlist_CR.xlsx",
                sheet_name='BadChans')
            #print(badchans)
            for o in range(block_num):
                h = o + 1
                badchan = badchans[str(h)].values
                badchan = badchan[badchan>0]
                #print(badchan)
                badchan = np.array(badchan[~np.isnan(badchan)] - 1, dtype='i4')  # matlab to python
                #print(badchan)

                stim_table = pd.read_excel(
                    self.path_patient + '/Data/experiment' + str(exp) + '/' + self.subj + "_stimlist_CR.xlsx",
                    sheet_name='Sheet' + str(h))
                # stim_table.insert(0, "Number", np.arange(len(stim_table)), True)
                stim_table = stim_table[stim_table.ChanP == ChanP]

                EEG_block = np.zeros((len(self.labels_all), len(stim_table), self.dur_tot * self.Fs))
                EEG_block[:, :, :] = np.NaN
                matfile = h5py.File(
                    self.path_patient + '/Data/experiment' + str(exp) + '/data_blocks/time/' + self.subj + "_BP_CR" + str(h) + "_pp/"+ self.subj + "_BP_CR" + str(h) + "_pp.mat", 'r')['EEGpp']
                EEGpp = matfile[()].T
                for s in range(len(stim_table)):
                    trig = stim_table.TTL_DS.values[s]
                    if np.int(trig + self.dur[0, 1] * self.Fs) > EEGpp.shape[1]:
                        EEG_block[:, s, 0:EEGpp.shape[1] - np.int(trig + self.dur[0, 0] * self.Fs)] = EEGpp[:, np.int(trig + self.dur[0, 0] * self.Fs):EEGpp.shape[1]]
                    elif np.int(trig + self.dur[0, 0] * self.Fs) < 0:
                        EEG_block[:, s, abs(np.int(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:,0:np.int(trig + self.dur[0, 1] * self.Fs)]
                    else:
                        EEG_block[:, s, :] = EEGpp[:, np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]
                if o == 0:
                    stim_list = stim_table
                else:
                    stim_list = pd.concat([stim_list, stim_table], sort=False)
                EEG_block[badchan, :, :] = np.nan
                EEG_stims = np.concatenate([EEG_stims, EEG_block], 1)

                #
            EEG_stims = np.delete(EEG_stims, (0), axis=1)
            stim_list.to_csv(self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day'+str(exp)+'.csv',index=False, header=True)
            np.save(self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day'+str(exp)+'.npy',EEG_stims)
            print('Data block saved \n')

    def cut_blocks_h(self, exp, block_num):
        for o in range(block_num):
            h = o + 1
            # stim_table    = pd.read_csv(path_patient+'/experiment'+str(exp)+'/InputOutput/' +subj+ "_stimlist_Ph.csv",sep=',',sheet_name='IO1')#
            stim_table = pd.read_excel(
                self.path_patient + '/experiment' + str(exp) + '/Circadian_PP/' + self.subj + "_stimlist_CR.xlsx",
                sheet_name='Sheet' + str(h))
            stim_table.insert(0, "Number", np.arange(len(stim_table)), True)

            EEG_block = np.zeros((len(self.labels_all), len(self.stim_table), self.dur_tot * self.Fs))
            EEG_block[:, :, :] = np.NaN
            matfile = h5py.File(
                self.path_patient + '/experiment' + str(exp) + '/data_blocks/time/' + self.subj + "_BP_CR" + str(h) + "_pp.mat",
                'r')['EEGpp']
            # Fs       = h5py.File(path_patient + '/experiment'+str(exp)+'/data_blocks/time/' + subj + "_BP_CR"+str(h)+"_pp.mat",'r')['Fs'][0,0]
            EEGpp = matfile[()].T
            for s in range(len(stim_table)):
                trig = stim_table.TTL_DS.values[s]
                # trig             = np.int(trig/4)
                if np.int(trig + dur[0, 1] * Fs) > EEGpp.shape[1]:
                    EEG_block[:, s, 0:EEGpp.shape[1] - np.int(trig + dur[0, 0] * self.Fs)] = EEGpp[:,np.int(trig + self.dur[0, 0] * self.Fs):EEGpp.shape[1]]
                elif np.int(trig + self.dur[0, 0] * self.Fs) < 0:
                    EEG_block[:, s, abs(np.int(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:, 0:np.int(trig + self.dur[0, 1] * self.Fs)]
                else:
                    EEG_block[:, s, :] = EEGpp[:, np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]

            np.save(self.path_patient + '/experiment' + str(exp) + '/Circadian_PP/All_resps_CR' + str(h) + '.npy', EEG_block)
            print('Data block saved', h)
    def get_LL_c(self, exps, w, sc, c):
        # w window in seconds
        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        data_LL         = np.zeros((1, 9))
        SP_LL_mean      = np.zeros((2, 1))
        print(StimChan)
        print(self.labels_all[c])
        for u in range(len(exps)):
            exp         = exps[u]
            #block_num   = block_nums[u]
            stim_table = pd.read_csv(self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day' + str(
                exp) + '.csv')  # scat_plot = scat_plot.fillna(method='ffill')
            stim_table.insert(0, "Number", np.arange(len(stim_table)), True)
            EEG_block = np.load(
                self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day' + str(exp) + '.npy')

            stim_list_spef  = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0)]  # &(stim_table.noise==0)
            stimNum         = stim_list_spef.Number.values
            IPIs            = np.array(stim_list_spef.IPI_ms.values)
            LL              = LL_funcs.get_LL_both(data=EEG_block[c:c + 1, stimNum, :], Fs=500, IPI=IPIs, t_0=5, win=w)
            val             = np.zeros((LL.shape[1], 9))
            val[:, 0]       = stim_list_spef.stim_block.values  # Stimblock number
            val[:, 1]       = stim_list_spef.h.values  # hour of day
            val[:, 2]       = stim_list_spef.Int_cond.values  # conidtioning intensity, mA
            val[:, 3]       = stim_list_spef.IPI_ms.values  # IPI, ms
            val[:, 4]       = LL[0, :, 0]  # LL of first pulse, abs
            val[:, 5]       = LL[0, :, 1]  # LL of second pulse, abs
            val[:, 7]       = exp  # day
            val[:, 6]       = stim_list_spef.sleep.values  # sleep state
            val[:, 8]       = stim_list_spef.StimNum.values  # sleep state

            ##normalize LL by  mean SP 2mA
            stim_list_spef = stim_table[(stim_table.ChanP == ChanP) & (stim_table.noise == 0) & (((
                        stim_table.Int_cond == 2) & (stim_table.IPI_ms > 120) ) | (stim_table.IPI_ms == 0))]
            stimNum         = stim_list_spef.Number.values
            IPIs            = np.zeros(stim_list_spef.IPI_ms.values.shape)
            LL_SP           = LL_funcs.get_LL_both(data=EEG_block[c:c + 1, stimNum, :], Fs=500, IPI=IPIs, t_0=5,
                                         win=w)  # data_LL     = LL[0,:,0]
            SP_LL_mean[u, 0] = np.nanmean(LL_SP[0, :, 1])
            data_LL = np.concatenate((data_LL, val), axis=0)

        data_LL = data_LL[1:-1, :]
        LL_all = pd.DataFrame(
            {"Block": data_LL[:, 0], "hour": data_LL[:, 1], "Int": data_LL[:, 2], "IPI": data_LL[:, 3],
             "LL SP": data_LL[:, 4], "LL PP": data_LL[:, 5], "Day": data_LL[:, 7], "Sleep": data_LL[:, 6],
             "StimNum": data_LL[:, 8]})
        # real SP
        LL_all.loc[LL_all['IPI'] == 0, 'LL SP'] = LL_all[LL_all['IPI'] == 0]['LL PP'].values
        for i in range(len(self.Int_all)):
            Int = self.Int_all[i]
            LL_all[LL_all['Int'] == Int] = LL_all[LL_all['Int'] == Int].fillna(method='ffill', axis=0)
            LL_all[LL_all['Int'] == Int] = LL_all[LL_all['Int'] == Int].fillna(method='bfill', axis=0)

        # l PP/SP ratio
        LL_all.insert(6, "LL PP/SP", (LL_all['LL PP'] / LL_all['LL SP']).values, True)
        # PP normalized (mean 2mA SP)
        LL_all.insert(7, "LL PP norm", (LL_all['LL PP'] / SP_LL_mean[0, 0]).values, True)
        LL_all.loc[(LL_all.Day == 2), 'LL PP norm'] = LL_all[(LL_all.Day == 2)]['LL PP'] / SP_LL_mean[1, 0]
        # SP normalized (mean 2mA SP)
        LL_all.insert(8, "LL SP norm", (LL_all['LL SP'] / SP_LL_mean[0, 0]).values, True)
        LL_all.loc[(LL_all.Day == 2), 'LL SP norm'] = LL_all[(LL_all.Day == 2)]['LL SP'] / SP_LL_mean[1, 0]
        # adding anatomical label
        # LL_all.insert(11, "Stim_anat", np.repeat(stim_cat[sc], LL_all.shape[0]), True)
        # LL_all.insert(12, "Resp_anat", np.repeat(resp_cat[c], LL_all.shape[0]), True)

        # saving
        LL_all.to_csv(self.path_patient + '/Analysis/Circadian_PP/data/LL/LL_' + StimChan + '-' + str(self.labels_all[c]) + '.csv',
            index=False, header=True)  # scat_plot = scat_plot.fillna(method='ffill')
        print('Data saved')
    def plot_stimNum_IPI_Int(self, sc, c, exp, IPI, Int):
        StimChan    = self.StimChans[sc]
        stim_table   = pd.read_csv(self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day'+str(exp)+'.csv')
        stim_table.insert(0, "Number", np.arange(len(stim_table)), True)
        EEG_block   = np.load(self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day'+str(exp)+'.npy')

        stim_list_spef  = stim_table[(stim_table.noise == 0) & (stim_table.IPI_ms == IPI) & (stim_table.Int_prob == Int)]  # &(stim_table.noise==0)
        stimNum         = stim_list_spef.Number.values

        resp        = np.nanmean(EEG_block[c, stimNum, :], 0)
        resp_std    = np.nanstd(EEG_block[c, stimNum, :], 0)
        plt.plot(self.x_ax, resp, lw=2.5, label='n = ' + str(len(stimNum)))
        plt.title(str(self.subj + ', experiment: '+str(exp)+", Stim: " + StimChan + ', Resp: ' + self.labels_all[c]))
        plt.fill_between(self.x_ax, resp - resp_std, resp + resp_std, alpha=0.1)
        plt.ylabel('uV', fontsize=18)
        plt.legend()
        plt.ylim([-800, 800])
        plt.xlim([-0.5, 2])
        plt.show()

    def plot_stimNum(self, sc, c, exp, stimNum):
        StimChan    = self.StimChans[sc]
        #stim_list   = pd.read_csv(self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day'+str(exp)+'.csv')
        EEG_block   = np.load(self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day'+str(exp)+'.npy')

        resp        = np.nanmean(EEG_block[c, stimNum, :], 0)
        resp_std    = np.nanstd(EEG_block[c, stimNum, :], 0)
        plt.plot(self.x_ax, resp, lw=2.5, label='n = ' + str(len(stimNum)), c = self.color_elab[0,:])
        plt.title(str(self.subj + ', experiment: '+str(exp)+", Stim: " + StimChan + ', Resp: ' + self.labels_all[c]))
        plt.fill_between(self.x_ax, resp - resp_std, resp + resp_std, alpha=0.1, color = self.color_elab[0,:])
        plt.ylabel('uV', fontsize=18)
        plt.legend()
        plt.ylim([-800, 800])
        plt.xlim([-0.5, 2])
        plt.xlabel('time [s]')
        plt.axvline(0, ls='--', c=[1,0,0], linewidth=3)
        #plt.show()

    def plot_stimNum_C(self, sc, chans, exp, stimNum, IPI):
        StimChan = self.StimChans[sc]
        # stim_list   = pd.read_csv(self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day'+str(exp)+'.csv')
        EEG_block = np.load(
            self.path_patient + '/Analysis/Circadian_PP/data/Resps_CR_' + StimChan + '_day' + str(exp) + '.npy')

        fig = plt.figure(figsize=(12, len(chans) * 3))
        gs = fig.add_gridspec(len(chans), 1)
        i=0
        for c in chans:
            resp        = np.nanmean(EEG_block[c, stimNum, :], 0)
            resp_std    = np.nanstd(EEG_block[c, stimNum, :], 0)
            ax = fig.add_subplot(gs[i, 0])  #
            plt.plot(self.x_ax, resp, lw=2.5, label='n = ' + str(len(stimNum)), c=self.color_elab[0, :])
            plt.title(str(self.subj + ', exp: ' + str(exp) + ", Stim: " + StimChan))
            plt.fill_between(self.x_ax, resp - resp_std, resp + resp_std, alpha=0.1, color=self.color_elab[0, :])
            plt.ylabel(self.labels_all[c], fontsize=18)
            plt.legend()
            plt.ylim([-1000, 1000])
            plt.xlim([-2, 4])
            plt.xticks([])
            plt.axvline(0, ls='--', c=[1, 0, 0], linewidth=3)
            plt.axvline(IPI/1000, ls='--', c=[1, 0, 0], linewidth=3)
            if self.labels_all[c] == StimChan:
                ax.set_facecolor('#f7cbcb')
            else:
                ax.set_facecolor('#fafafa')

            i=i+1

        plt.xticks([-2,-1,0,1,2,3,4])
        plt.xlabel('time [s]')

    # plt.show()

    def get_hypnogram(self,exps):
        ## Day 1
        for u in range(len(exps)):
            exp = exps[u]
            block_num = block_nums[u]
            for i in range(block_num):
                h = i + 1
                stim_table = pd.read_excel(
                    path_patient + '/experiment' + str(exp) + '/Circadian_PP/' + subj + "_stimlist_CR.xlsx",
                    sheet_name='Sheet' + str(h))
                # stim_table.insert(0, "Number", np.arange(len(stim_table)), True)
                if h == 1:
                    hypnogram1 = stim_table
                else:
                    hypnogram1 = pd.concat([hypnogram1, stim_table])

    def plot_LL_IPI(self, sc, c):
        w=0.1
        StimChan    = self.StimChans[sc]
        ChanP       = self.StimChanNums[sc]
        print(StimChan)
        print(self.labels_all[c])
        LL_all = pd.read_csv(
            self.path_patient + '/Analysis/Circadian_PP/data/LL/LL_' + StimChan + '-' + str(self.labels_all[c]) + '.csv')

        ## norm LL over IPI
        plt.style.use('seaborn-colorblind')  # sns.set_style("whitegrid")
        # sns.set(font_scale=2)
        sns.catplot(x="IPI", y="LL PP norm", hue="Int", data=LL_all[(LL_all['Int'] > 0)], kind="swarm",
                    height=5,  # make the plot 5 units high
                    aspect=2.5, s=5, legend_out=True)  # .set(ylim=(0, 3000))  # length is 4x heigth
        plt.ylim([0, 3])
        plt.suptitle(str('LL ['+str(w)+'s]--- '+ self.subj +", Stim: " + StimChan + ', Resp: ' + self.labels_all[c]), fontsize=20,y=1)
        plt.xticks([0, 6, 13, 18, 22, 26, 28], ['10', '30', '100', '250', '500', '1000', '1600'])
        plt.xlabel('IPI [ms]', fontsize=18)
        plt.ylabel('LL norm', fontsize=18)
        plt.tick_params(axis='both', labelsize=18)
        plt.axhline(1, ls='--', c=[0, 0, 0], linewidth=1.5)
        # plt.savefig(path_patient + '/Analysis/Circadian_PP/PP/LL_norm_Sleep_'+StimChan+ '-'+str(labels_all[c])+'.jpg')
        plt.show()