import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.fftpack
from scipy.signal import find_peaks
import scipy.io as sio
import freq_funcs as ff
import LL_funcs as LLf
# Importing libraries
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels
import significance_funcs as sf
import LL_funcs
cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]


####GENERAL functions to analyze characteristics of a CCEP
# Function to calculate SNR
def update_sig_FDR(con_trial):
    req = (con_trial.Artefact<1)
    p_val = con_trial.loc[req, 'p_value_LL']
    p_val = abs(1-p_val)
    p_sig, p_corr = statsmodels.stats.multitest.fdrcorrection(p_val)
    p_sig*1
    con_trial.loc[req, 'Sig']

def calculate_snr(sc, rc, con_trial, EEG_resp, win_LL = 0.25, win_p2p = 0.5, t0 = 1, Fs = 500):
    n_surr = 50
    table = []
    for sig_value in [0,1]:
         # todo add surrogates:
        trial = con_trial.loc[(con_trial.Stim==sc)&(con_trial.Chan==rc)&(con_trial.Sig== sig_value)&(con_trial.Artefact<1), 'Num'].values
        if len(trial)>0:
            trial_surr = np.unique(con_trial.loc[(con_trial.Stim!=sc)&(con_trial.Chan==rc)&(con_trial.Artefact<1), 'Num'].values)
            surr_val = np.zeros((n_surr, 2))
            t_test = t0-0.6
            for i in range(n_surr):
                trial_surr_sel = np.unique(np.random.choice(trial_surr, len(trial)))
                signal = np.mean(EEG_resp[rc, trial_surr_sel], axis=0)
                surr_val[i,0] = np.ptp(signal[int((t_test)*Fs):int((t_test+win_p2p)*Fs)])
                LLt = LL_funcs.get_LL_all(np.expand_dims(signal[int((t_test-win_LL/2)*Fs):int((t_test+0.5+win_LL/2)*Fs)], [0, 1]), Fs, 0.25)[0, 0]
                surr_val[i,1] = np.max(LLt[int(win_LL/2*Fs):-int(win_LL/2*Fs)])

            signal = np.mean(EEG_resp[rc, trial], axis=0)
            for t_test, t_test_val in zip([t0-0.6, t0], [0,1]):
                p2p  = np.ptp(signal[int((t_test)*Fs):int((t_test+win_p2p)*Fs)])
                LL = LL_funcs.get_LL_all(np.expand_dims(signal[int((t_test-win_LL/2)*Fs):int((t_test+0.5+win_LL/2)*Fs)], [0, 1]), Fs, 0.25)[0, 0]
                LL = np.max(LL[int(win_LL/2*Fs):-int(win_LL/2*Fs)])
                p2p_snr = (p2p-np.mean(surr_val[:,0]))/np.std(surr_val[:,0])
                LL_snr = (LL-np.mean(surr_val[:,1]))/np.std(surr_val[:,1])
                table.append([sc, rc, sig_value, len(trial), t_test_val, p2p, LL,p2p_snr, LL_snr])
    return table