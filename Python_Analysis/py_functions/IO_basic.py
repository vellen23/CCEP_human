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
import basic_func as bf


def get_LL_all(EEG_resp, stimlist, bad_chans, path_patient, w=0.25):
    data_LL                     = np.zeros((1, 10))  # RespChan, Int, LL, LLnorm, State
    Int_all                     = np.unique(stimlist.Int_prob)
    stim_spec                = stimlist[(stimlist.noise ==0)]#&(stimlist.noise ==0)
    stimNum                  = stim_spec.StimNum.values#[:,0]
    resps                    = ff.lp_filter(EEG_resp[:,stimNum,:],45,Fs)
    ChanP1                   = bf.SM2IX(stim_spec.ChanP.values,StimChanSM,np.array(StimChanIx))
    IPIs                     = np.expand_dims(np.array(stim_spec.IPI_ms.values),1)
    LL                = LL_funcs.get_LL_both(data=resps, Fs=Fs, IPI=IPIs, t_0=1, win=w)
    pk_start = 0.5
    for c in range(len(LL)):
        val         = np.zeros((LL.shape[1], 10))
        val[:, 0]   = c                                         # response channel
        val[:, 1]    = ChanP1
        val[:, 4]   = stim_spec.Int_prob.values              # Intensity
        val[:, 2]   = LL[c, :, 1]                               # PP
        val[:, 6]   = stim_spec.condition.values
        val[:, 7]   = stimNum
        val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic)==1),2] = np.nan

        pks         = np.max(abs(resps[c,:,np.int64(pk_start*Fs):np.int64(1.5*Fs)]),1)
        pks_loc     = np.argmax(abs(resps[c,:,np.int64(pk_start*Fs):np.int64(1.5*Fs)]),1)+np.int64(pk_start*Fs)
        #ix         = np.where(np.max(abs(resps[c,:,np.int64(0.95*Fs):np.int64(1.01*Fs)]),1)>400)
        ix          = np.where((pks>100)&(pks_loc>np.int64(0.95*Fs))&(pks_loc<np.int64(1.005*Fs)))
        val[ix, 2]  = np.nan

        data_LL     = np.concatenate((data_LL, val), axis=0)

    data_LL = data_LL[1:-1, :] # remove first row (dummy row)
    LL_all = pd.DataFrame(
        {"Chan": data_LL[:, 0],"Stim": data_LL[:, 1], "LL": data_LL[:,2], "nLL": data_LL[:, 2], "Int": data_LL[:, 4],
         "Condition": data_LL[:, 6],"Num": data_LL[:, 7]})
    # define condition (benzo, bl, sleepdep)
    LL_all.insert(7, "Cond", 0, True)
    for j in range(len(cond_vals)):
        LL_all.loc[(LL_all.Condition == cond_vals[j]), 'Cond'] = cond_labels[j]
    LL_all.loc[(LL_all.Chan).isin(bad_chans), 'LL'] = np.nan
    #LL_all.insert(2, 'Sig_Con', 0)
    # distance
    for s in np.unique(LL_all.Stim):
        s   = np.int64(s)
        #resps_chan= np.where(sig_chan.values[s]==1)[0]
        #LL_all = LL_all.drop(LL_all[(LL_all.Stim == s)&(~(LL_all.Chan.isin(resps_chan)))].index)
        #LL_all.loc[(LL_all.Stim == s)&((LL_all.Chan.isin(resps_chan))),'Sig_Con' ] = 1
        for c in np.unique(LL_all.Chan):
            c   = np.int64(c)
            mx = np.nanmean(LL_all.loc[(LL_all.Stim == s)&(LL_all.Chan == c)&(LL_all.Condition == 1)&(LL_all.Int == np.max(Int_all)), 'LL'])
            LL_all.loc[(LL_all.Stim == s)&(LL_all.Chan == c), 'd'] = math.sqrt(((coord_all[s,0]-coord_all[c,0])**2)+((coord_all[s,1]-coord_all[c,1])**2)+((coord_all[s,2]-coord_all[c,2])**2))
            LL_all.loc[(LL_all.Stim == s)&(LL_all.Chan == c), 'nLL'] = LL_all.loc[(LL_all.Stim == s)&(LL_all.Chan == c), 'LL'] / mx
    LL_all.to_csv(path_patient + '/Analysis/InputOutput/Ph/data/LL_CCEP.csv', index=False,header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    return LL_all

def get_LL_thr(EEG_resp, LL_all, path_patient, num_trials=3):
    ## get threshoold value for each response channel (99th and 95h)
    chan_thr        = np.zeros((len(EEG_resp), 4))
    for rc in range(len(EEG_resp)):
        chan_thr[rc,:] = bf.get_sig_thr(rc, LL_all, EEG_resp, num_trials)
    data_A = pd.DataFrame(chan_thr, columns=['99', '95', '90', 'std'])
    data_A.to_csv(path_patient + '/Analysis/InputOutput/Ph/data/chan_sig_thr.csv', index=False,header=False)  # scat_plot = scat_plot.fillna(method='ffill')
    return chan_thr

def get_IO_LL_Int(EEG_resp, LL_all, chan_thr, path_patient, type = 'Condition'):
    # type: #Condition (Ph) or Sleep (CR)
    # calculates for each channel pair and intensity the mean CCEP and LL and checks whether it is significant
    LL_mean       = np.zeros((1,8))
    lists         = LL_all[(LL_all['Int']>0)]
    lists         = lists[~np.isnan(lists.LL.values)]
    stims         = np.unique(lists.Stim)
    Int_selc      = np.unique(lists.Int)
    cond_sel      = np.unique(lists[type])

    for sc in stims:
        sc    = np.int64(sc)
        resps = np.unique(lists.loc[(lists.Stim==sc), 'Chan'])
        for rc in resps:
            rc           = np.int64(rc)
            for j in range(len(cond_sel)): # each condition
                for i in range(len(Int_selc)): # each intensity
                    dati = lists[(lists.Int==Int_selc[i])&(lists.Stim==sc)&(lists.Chan==rc)&(lists[type]==cond_sel[j])]
                    if len(dati)>0:
                        resp = np.nanmean(EEG_resp[rc,dati.Num.values.astype('int'),: ], 0)
                        LL_resp,mx,mx_ix, sig  = bf.sig_resp(resp, chan_thr[rc,1])

                        val         = np.zeros((1,8))
                        val[0, 0]   = rc                                         # response channel
                        val[0, 1]   = sc                                         # response channel
                        val[0, 2]   = mx
                        val[0, 3]   = Int_selc[i]
                        val[0, 4]   = np.nanmean(dati.d)
                        val[0, 5]   = sig
                        val[0, 6]   = cond_sel[j]
                        #val[0, 7]   = np.nanmean(dati['Sig_Con'])

                        LL_mean   = np.concatenate((LL_mean, val), axis=0)
    LL_mean = LL_mean[1:-1, :] # remove first row (dummy row)
    data_A = pd.DataFrame(
        {"Chan": LL_mean[:, 0], "Stim": LL_mean[:, 1], "LL": LL_mean[:, 2], "Sig": LL_mean[:, 5], "Int": LL_mean[:, 3], "d": LL_mean[:, 4], type: LL_mean[:, 6]}) #, "Sig_Con": LL_mean[:, 7]
    for sc in stims:
        sc          = np.int64(sc)
        resps = np.unique(data_A.loc[(data_A.Stim==sc), 'Chan'])
        for rc in resps:
            data_A.loc[(data_A.Stim==sc)&(data_A.Chan==rc), 'LL norm']= data_A.loc[(data_A.Stim==sc)&(data_A.Chan==rc), 'LL'] / data_A.loc[(data_A.Int==np.max(Int_selc))&(data_A[type]==cond_sel[0])&(data_A.Stim==sc)&(data_A.Chan==rc), 'LL'].values[0]
    #for c in range(len(labels_all)):
    #    data_A.loc[(data_A.Chan ==c), "Recs"]   = labels_all[c]
    #    data_A.loc[(data_A.Stim ==c), "Stim Region"]   = labels_region[c]
    #    data_A.loc[(data_A.Chan ==c), "Resp Region"]   = labels_region[c]
    #    data_A.loc[(data_A.Stim ==c), "Stims"]  = labels_all[c]
    #data_A=data_A.drop(data_A[data_A['Stim Region']=='OUT'].index)
    #data_A=data_A.drop(data_A[data_A['Resp Region']=='OUT'].index)
    if type == 'Condition':
        data_A.to_csv(path_patient + '/Analysis/InputOutput/Ph/data/sig_intensity.csv', index=False,header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    elif type =='Sleep':
        data_A.to_csv(path_patient + '/Analysis/InputOutput/CR/data/sig_intensity.csv', index=False, header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    else:
        data_A.to_csv(path_patient + '/Analysis/InputOutput/data/sig_intensity.csv', index=False, header=True)  # scat_plot = scat_plot.fillna(method='ffill')

    return data_A

def get_IO_summary(LL_mean, EEG_resp,path_patient, type = 'Condition' ):
    data_mean     = np.zeros((1,8))
    data_test     = LL_mean[LL_mean.LL>0] # no artefacts
    stims         = np.unique(data_test.Stim)
    Int_all       = np.unique(data_test.Int)
    for sc in stims: # repeat for each stimulation channel
        sc    = np.int64(sc)
        resps = np.unique(data_test.loc[(data_test.Stim==sc), 'Chan'])
        for rc in resps:
            rc       = np.int64(rc)
            LL0      = np.min((data_test.loc[(data_test.Stim==sc)&(data_test.Chan==rc), 'LL norm']).values)
            cond_sel = np.unique(data_test[type])
            for j in range(len(cond_sel)):
                dati        = data_test[(data_test.Stim==sc)&(data_test.Chan==rc)&(data_test[type]==cond_sel[j])]
                if len(dati)>0:
                    val         = np.zeros((1, 8))
                    val[0, 0]   = rc                                         # response channel
                    val[0, 1]   = sc
                    val[0, 6]   = cond_sel[j] #condition
                    val[0, 4]   = np.nanmean(dati.d) # distance
                    val[0, 2]   = trapz(dati['LL norm'].values- LL0, dati['Int'].values) #AUC
                    Int_min     = np.unique(dati.loc[dati.Sig==1, 'Int'])
                    if len(np.unique(dati.loc[dati.Sig==0, 'Int']))>0:
                        Int_min     = np.unique(dati.loc[dati.Sig==1, 'Int'])[np.where(Int_min -np.unique(dati.loc[dati.Sig==0, 'Int'])[-1]>0)]
                    if (len(Int_min)>0) and (np.mean(dati.loc[dati.Int>Int_all[-3], 'Sig'])>=0.5) :
                        Int_min = Int_min[0]
                        #Int_min = np.unique(dati.loc[dati.Sig==1, 'Int'])[np.where(Int_min -np.unique(dati.loc[dati.Sig==0, 'Int'])[-1]>0)[0].astype('int')[0]]

                        #val[0, 2]   = dati.loc[dati.Int == Int_min, 'LL'].values
                        val[0, 3]   = Int_min
                        val[0, 5]   = dati.loc[dati.Int == Int_min, 'LL norm'].values
                        val[0, 7]   = 1
                        data_mean   = np.concatenate((data_mean, val), axis=0)
                    else:
                        Int_min     = 0
                        #val[0, 2]   = 0
                        val[0, 3]   = Int_min
                        val[0, 5]   = 1
                        val[0, 7]   = 0


    data_mean = data_mean[1:-1, :] # remove first row (dummy row)
    IO_mean = pd.DataFrame(
        {"Chan": data_mean[:, 0], "Stim": data_mean[:, 1], "AUC": data_mean[:, 2], "MPI": data_mean[:, 3], "d": data_mean[:, 4], type: data_mean[:, 6], "Sig": data_mean[:, 7]})
    if type =='Condition':
        path_file = path_patient + '/Analysis/InputOutput/Ph/data/IO_mean.csv'
    elif type =='Sleep':
        path_file = path_patient + '/Analysis/InputOutput/CR/data/IO_mean.csv'
    else:
        path_file = path_patient + '/Analysis/InputOutput/data/IO_mean.csv'
    IO_mean.to_csv(path_file, index=False,header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    print('Saving IO_mean:     '+ path_file)
    return IO_mean

def plot_raw_LL_IO(sc, rc, LL_all, LL_mean, EEG_resp, labels_all,  path_patient, type = 'Condition'):
    dat      = LL_all[(LL_all['Stim']==sc)&(LL_all['Chan']==rc)]
    conds    = np.unique(dat[type])
    w        = 0.25
    fig, axs = plt.subplots(len(conds),3, figsize=(15, 8), facecolor='w', edgecolor='k')
    axs     = axs.ravel()
    plt.close(fig) # todo: find better solution
    fig      = plt.figure(figsize=(15, 8), facecolor='w', edgecolor='k')
    #
    gs       = fig.add_gridspec(len(conds),3,width_ratios=[1,1,2])  # GridSpec(4,1, height_ratios=[1,2,1,2])
    for i in range(len(conds)):
        axs[2*i+0] = fig.add_subplot(gs[i, 0])
        axs[2*i+1] = fig.add_subplot(gs[i, 1])

    axIO   = fig.add_subplot(gs[:, 2])
    plt.suptitle(subj+' -- '+labels_all[np.int64(sc)]+', Resp: '+labels_all[np.int64(rc)]+', d='+str(np.round(np.mean(dat.d),1))+'mm', y=0.95)
    limy_LL          = 3 # limits for LL plot
    limy_CCEP        = 200
    Int_selc         = np.unique(dat.Int)
    colors_Int       = np.zeros((len(Int_selc), 3))
    colors_Int[:, 0] = np.linspace(0, 1, len(Int_selc))
    LL0              = np.min((LL_mean.loc[(LL_mean.Stim==sc)&(LL_mean.Chan==rc), 'LL norm']).values)
    mx_LL =1
    for j in range(len(conds)):
        con_sel = np.int64(conds[j])
        Int_selc = np.unique(dat.loc[(dat.Stim==sc)&(dat.Chan==rc)&(dat[type]==con_sel), 'Int'])
        for i in range(len(Int_selc)):
            dati = dat[(dat.Int==Int_selc[i])&(dat.Stim==sc)&(dat.Chan==rc)&(dat[type]==con_sel)]
            if len(dati)>0:
                resp             = np.nanmean(EEG_resp[rc,dati.Num.values.astype('int'),: ], 0)
                LL_resp,mx,max_ix, sig  = sig_resp(resp, chan_thr[rc,1])

                axs[0+2*j].plot( x_ax,resp, c = colors_Int[i], alpha=0.5+0.5*sig, linewidth= 1.2*(0.5+sig))
                axs[0+2*j].set_xlim(-0.2, 0.5)
                axs[1+2*j].plot( x_ax,LL_resp, c = colors_Int[i], alpha=0.5+0.5*sig, linewidth= 1.2*(0.5+sig))
                axs[1+2*j].plot(0.01+w/2+max_ix/Fs,mx, marker='+', c = [0,0,0], alpha=0.7+0.3*sig, markersize= 10)
                axs[1+2*j].set_xlim(-0.2, 0.5)
                #axIO.plot(Int_selc[i], mx, marker='o', markersize=10, c = cond_colors[con_sel], alpha=0.2+0.8*sig)
                mx_norm = LL_mean.loc[(LL_mean.Stim==sc)&(LL_mean.Chan==rc)&(LL_mean.Int==Int_selc[i])&(LL_mean[type]==con_sel), 'LL norm']
                axIO.plot(Int_selc[i], mx_norm, marker='o', markersize=10, c = cond_colors[con_sel], alpha=0.2+0.8*sig)
                limy_LL = np.nanmax([limy_LL, mx])
                limy_CCEP = np.nanmax([limy_CCEP, np.max(abs(resp))])
                mx_LL = np.max([mx_LL, mx_norm.values[0]])
        #y = (data_A.loc[(data_A.Stim==sc)&(data_A.Chan==rc)&(data_A[type]==con_sel), 'LL norm']*data_A.loc[(data_A.Stim==sc)&(data_A.Chan==rc)&(data_A[type]==con_sel), 'Sig']).values
        y = (LL_mean.loc[(LL_mean.Stim==sc)&(LL_mean.Chan==rc)&(LL_mean[type]==con_sel), 'LL norm']).values- LL0

        axIO.plot(Int_selc[i], mx_norm, marker='o', markersize=10, c = cond_colors[con_sel], alpha=0.2+0.8*sig, label= cond_labels[con_sel]+', AUC: '+str(np.round(trapz(y, Int_selc),2)))

    for i in range(len(conds)*2):
        axs[i].axvline(0, c=[0,0,0])


    axIO.legend(loc='lower right')
    axIO.set_title('IO curve')
    axIO.set_ylabel('LL uv/ms [250ms] normalized')
    axIO.set_xlabel('Intensity [mA]')
    axIO.set_ylim([0,1.1*mx_LL])
    axIO.axhline(LL0, color="black", linestyle="--")

    axs[0].set_title('mean CCEP')
    axs[1].set_title('LL ['+str(w)+'s] of mean CCEP')
    for i in range(len(conds)):
        axs[2*i].set_ylabel(cond_labels[np.int64(conds[i])])
        axs[2*i].set_ylim(-limy_CCEP,limy_CCEP)
        #axs[2*i].axvspan(0.01, 0.01+w, alpha=0.05, color='blue')
    for i in range(len(conds)):
        #axs[2*i+1].axvline(0.01+w/2, c=[0,0,1], alpha = 0.5)
        axs[2*i+1].set_ylim(0,1.2*limy_LL)
    axs[2].set_xlabel('time [s]')
    axs[3].set_xlabel('time [s]')

    plt.savefig(path_patient + '/Analysis/InputOutput/Ph/figures/'+subj+'_IO_'+labels_all[sc]+'_'+labels_all[rc]+'.jpg')
    plt.show()