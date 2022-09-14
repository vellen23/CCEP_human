import numpy as np
import copy
from scipy import signal
from scipy.stats import zscore
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import sys
sys.path.append('./py_functions')
import freq_funcs as ff


def get_notch_picks(x, fs, show=False):
    f, Pxx_den      = signal.welch(x, fs, nperseg=1*fs)
    if show:
        for i in range(len(Pxx_den)):
            plt.semilogy(f, Pxx_den[i, :])
        # plt.ylim([0.5e-3, 1])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()
    picks           = np.zeros((len(Pxx_den)))
    for c in range(len(Pxx_den)):
        thr         = np.mean(Pxx_den[c,60:95])
        peaks, _    = find_peaks(Pxx_den[c,:], height=thr)
        if np.sum(peaks == 100):
            picks[c] = 1
    pick_list = np.where(picks==1)
    return pick_list

def get_hgp(raw, freq_l=80, freq_h=150):
    # raw     = mne.io.read_raw_fif(path+"/mne_raw.fif", preload=1)
    n_steps = 20

    # Employ filtering with logarithmic spacing of filters
    freqs = np.geomspace(freq_l, freq_h, num=n_steps + 1, endpoint=True)
    steps_v = np.diff(freqs)

    # Initialize array:
    # raw_hilb    = np.zeros((raw._data.shape))
    # Initialize array:
    pwr = np.zeros((len(freqs) - 1, raw._data.shape[0], raw._data.shape[1]))
    i = 0
    # For each sub-frequency band:
    for f, stepf in zip(freqs, steps_v):
        # Make a copy of the raw unfiltered data
        raw2filt = copy.deepcopy(raw)

        # Filter the data
        raw2filt.filter(f, f + stepf)

        # Compute the hilbert transform and keep the envelope of the signal
        raw2filt.apply_hilbert(envelope=True)

        # Z-score each frequency band and then add up all frequencies
        # raw_hilb = raw_hilb + zscore(raw2filt._data, axis=1)
        pwr[i] = raw2filt._data
        i = i + 1
        del raw2filt
    #pwr = ff.lp_filter(pwr, 80, fs, order=5)
    #pwr = ff.resample(pwr, )
    # np.save(path+"/HGP.npy", pwr)
    return pwr

def get_sop(raw):
    # get slow oscillation power (delta, theta, alpha, beta)

    # Employ filtering with logarithmic spacing of filters
    freqs       = np.array([0.5,4, 8, 12, 30])
    steps_v     = np.diff(freqs)

    # Initialize array:
    # raw_hilb    = np.zeros((raw._data.shape))
    # Initialize array:
    pwr = np.zeros((len(freqs) - 1, raw._data.shape[0], raw._data.shape[1]))
    i = 0
    # For each sub-frequency band:
    for f, stepf in zip(freqs, steps_v):
        # Make a copy of the raw unfiltered data
        raw2filt        = copy.deepcopy(raw)
        # Filter the data
        raw2filt.filter(f, f + stepf)

        # Compute the hilbert transform and keep the envelope of the signal
        raw2filt.apply_hilbert(envelope=True)

        # Z-score each frequency band and then add up all frequencies
        # raw_hilb = raw_hilb + zscore(raw2filt._data, axis=1)
        pwr[i] = raw2filt._data
        i = i + 1
        del raw2filt

    # np.save(path+"/HGP.npy", pwr)
    return pwr

def load_hgp(hgp_path, freq_l=80, freq_h=150, raw=None):

    n_steps = 21

    # Employ filtering with logarithmic spacing of filters
    freqs   = np.geomspace(freq_l, freq_h, num=n_steps, endpoint=False)
    steps_v = np.diff(freqs)

    # Initialize array:
    raw_hilb = np.zeros((raw._data.shape))

    # For each sub-frequency band:
    for f, stepf in zip(freqs, steps_v):
              
        # Make a copy of the raw unfiltered data
        raw2filt = copy.deepcopy(raw)
        
        # Filter the data
        raw2filt.filter(f, f+stepf)
        
        # Compute the hilbert transform and keep the envelope of the signal
        raw2filt.apply_hilbert(envelope=True)
      
        # Z-score each frequency band and then add up all frequencies
        raw_hilb = raw_hilb + zscore(raw2filt._data, axis=1)
      
        del raw2filt

    # Take the mean across all frequency bands
    raw_hilb = np.divide(raw_hilb,len(freqs))

    # Replace the raw EEG
    raw._data = raw_hilb

    # Save
    raw.save(hgp_path, overwrite=True)
  
    del raw_hilb