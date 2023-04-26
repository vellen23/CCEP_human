import os
import numpy as np
import mne
import h5py
import pandas as pd
import sys

sys.path.append('./py_functions')
import hgp_func
from tkinter import *
import ntpath
from pandas import read_excel, ExcelWriter

root = Tk()
root.withdraw()
import copy
from scipy.io import savemat
import scipy
import platform
from glob import glob

# import SM_converter as SMC
cwd = os.getcwd()


class main:
    def __init__(self, subj, path_patient, dur=np.array([-1, 3])):

        #if not os.path.exists(path_gen):
        #    path_gen = 'T:\\EL_experiment\\Patients\\' + subj  # if not in hulk check in T drive
        # path_patient = path_gen + '\Data\EL_experiment'  # path where data is stored
        path_infos = os.path.join(path_patient, 'infos')  # infos is either in Data or in general
        if not os.path.exists(path_infos):
            print('Cant find infos folder in patients folder')
            # path_infos = path_gen + '\\infos'

        #  basics, get 4s of data for each stimulation, [-2,2]s
        self.Fs = 500
        self.dur = np.zeros((1, 2), dtype=np.int32)
        self.dur[0, :] = dur  # [-1, 3]
        self.dur_tot = np.int32(np.sum(abs(self.dur)))
        self.x_ax = np.arange(self.dur[0, 0], self.dur[0, 1], (1 / self.Fs))

        # load patient specific information
        lbls = pd.read_excel(os.path.join(path_patient, 'Electrodes', subj + "_labels.xlsx"), header=0, sheet_name='BP')
        if "type" in lbls:
            lbls = lbls[lbls.type == 'SEEG']
            lbls = lbls.reset_index(drop=True)
        self.labels = lbls.label.values
        self.labels_C = lbls.Clinic.values

        self.coord_all = np.array([lbls.x.values, lbls.y.values, lbls.z.values]).T
        # only healthy channels
        # tissue  = lbls[req].Tissue.values
        self.subj = subj
        self.path_patient = path_patient
        self.path_patient_analysis = os.path.join('Y:\eLab\EvM\Projects\EL_experiment\Analysis\Patients', subj)
        self.path_patient_analysis = os.path.join('X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients', subj)

    def osc_power(self, path, hgp=True, sop=True):
        try:
            matfile = h5py.File(path + "/ppEEG.mat", 'r')['ppEEG']
            EEGpp = matfile[()].T
        except IOError:
            EEGpp = scipy.io.loadmat(path + "/ppEEG.mat")['ppEEG']

        inf = mne.create_info(ch_names=self.labels.tolist(), sfreq=self.Fs, ch_types='seeg', verbose=None)
        raw = mne.io.RawArray(EEGpp, inf, first_samp=0, copy='auto', verbose=None)
        raw.info['lowpass'] = 200
        raw.info['highpass'] = 0.5
        # pick_list               = hgp_func.get_notch_picks(raw._data, self.Fs,False)
        # raw2filt                = copy.deepcopy(raw)
        # raw.filter(0.5, 200, fir_design='firwin')
        # if len(pick_list[0])>0:
        #     raw.notch_filter(np.arange(50, 241, 50), filter_length='auto', picks=pick_list[0], phase='zero')
        # mdic                    = {"fs": 500.0, "ppEEG": raw._data}
        # savemat(path + "/ppEEG.mat", mdic)
        # mdic = {"fs": 500.0, "ppEEG": EEGpp}
        # savemat(path + "/ppEEG0.mat", mdic)
        # print(ntpath.basename(path)+' ---- pp Saved')

        if hgp:
            if os.path.isfile(path + "/HGP.npy"):
                print('HGP already cal')
            else:
                print('calculate HGP')
                pwr = hgp_func.get_hgp(raw)
                np.save(path + "/HGP.npy", pwr)
        if sop:
            if os.path.isfile(path + "/SOP.npy"):
                print('SOP already cal')
            else:
                print('calculate SOP')
                pwr = hgp_func.get_sop(raw)
                np.save(path + "/SOP.npy", pwr)

    def filter_raw(self, path):
        try:
            matfile = h5py.File(path + "/EEG_art.mat", 'r')['EEG_art']
            EEGpp = matfile[()].T
        except IOError:
            EEGpp = scipy.io.loadmat(path + "/EEG_art.mat")['EEG_art']
        Fs = 2000
        inf = mne.create_info(ch_names=self.labels.tolist(), sfreq=Fs, ch_types='seeg', verbose=None)
        raw = mne.io.RawArray(EEGpp, inf, first_samp=0, copy='auto', verbose=None)

        pick_list = hgp_func.get_notch_picks(raw._data, Fs, False)
        raw2filt = copy.deepcopy(raw)
        raw2filt.filter(0.5, 200, fir_design='firwin')
        if len(pick_list[0]) > 0:
            raw2filt.notch_filter(np.arange(50, 241, 50), filter_length='auto', picks=pick_list[0], phase='zero')
        raw2filt.resample(500)
        mdic = {"fs": 500.0, "ppEEG": raw2filt._data}

        savemat(path + "/ppEEG_mne.mat", mdic)
        # mdic = {"fs": 500.0, "ppEEG": EEGpp}
        # savemat(path + "/ppEEG0.mat", mdic)
        print(ntpath.basename(path) + ' ---- pp Saved')

    def SM2_conv(self, path, block, type):
        # infos, always the same
        if type == 'PP':
            types = ['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP']
            folder = 'PairedPulse'
        elif type == 'IO':
            types = ['IO', 'Ph_IO', 'CR_IO', 'InputOutput']
            folder = 'InputOutput'
        elif type == 'BM':
            types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
            folder = 'BrainMapping'
        else:
            types = []
            folder = 'nofolder'
        if len(types) > 0:
            # Patient specific
            filename = ntpath.basename(path)
            data_path = os.path.dirname(os.path.dirname(path))
            subj = filename[0:5]  # EL000
            # data
            path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))
            a = 0
            t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
            if filename[-1].isnumeric():
                if filename[-2].isnumeric():
                    t = filename[9:-2]
                    p = int(filename[-2:])
                else:
                    t = filename[9:-1]
                    p = int(filename[-1])
                a = 1
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                           sheet_name='Sheet' + str(p))  #
            else:
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
            print(t)
            condition = t[0:2]
            stim_table = stim_table[stim_table['type'].isin(types)]
            conv = SMC.get_converter(subj)
            # stim_table = stim_table.drop(columns="Num", errors='ignore')
            ChanP0 = stim_table.ChanP
            ChanN0 = stim_table.ChanN
            stim_table.insert(10, "ChanP0", ChanP0, True)
            stim_table.insert(10, "ChanN0", ChanN0, True)
            ChanP = np.copy(ChanP0)
            ChanN = np.copy(ChanN0)
            for row in conv:
                value_SM2 = row[1]
                value_SM1 = row[0]
                ChanP[ChanP0 == value_SM2] = value_SM1
                ChanN[ChanN0 == value_SM2] = value_SM1
            stim_table.ChanP = ChanP
            stim_table.ChanN = ChanN

            # save as new sheet

            with ExcelWriter(data_path + "/" + subj + "_stimlist_" + t + ".xlsx", mode='a') as writer:

                if a == 1:
                    workBook = writer.book
                    try:
                        workBook.remove(workBook['Sheet' + str(p)])
                    except:
                        print("Worksheet does not exist")
                    stim_table.to_excel(writer, sheet_name='Sheet' + str(p), index=False)
                else:
                    workBook = writer.book
                    try:
                        workBook.remove(workBook['Sheet1'])
                    except:
                        print("Worksheet does not exist")
                    stim_table.to_excel(writer, sheet_name='Sheet1', index=False)

        else:
            print('ERROR: no type defined (BM, IO, PP)')

    def cut_resp_LT(self, path, path_save):
        types = ['LTD1', 'LTD10', 'LTP50']
        folder = 'LongTermInduction'

        # Patient specific
        filename = ntpath.basename(path)
        data_path = os.path.dirname(os.path.dirname(path))
        subj = filename[0:5]  # EL000

        t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
        if filename[-1].isnumeric():
            if filename[-2].isnumeric():
                t = filename[9:-3]
                p = int(filename[-2:])
            else:
                t = filename[9:-1]
                p = int(filename[-1])
            if t[-1] == '_':
                t = t[:-1]
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                       sheet_name='Sheet' + str(p))  #
        else:
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
            p = 0
        file_block = path_save + '\\' + folder + '/data/All_resps_' + str(p).zfill(
                2) + '_' + t + '.npy'

        if not os.path.exists(path_save + '\\' + folder):
            os.makedirs(path_save + '\\' + folder)
            os.makedirs(path_save + '\\' + folder + '\\data')

        stim_table = stim_table.drop(columns="Num", errors='ignore')
        stim_table = stim_table.reset_index(drop=True)
        stim_table.insert(10, "Num", np.arange(0, len(stim_table), True))
        if len(stim_table) > 0:
            if not os.path.exists(path_save + '\\' + folder + '/data/'):
                os.makedirs(path_save + '\\' + folder + '/data/')
            # Get bad channels
            if os.path.isfile(path + "/bad_chs.mat"):
                try:  # load bad channels
                    matfile = h5py.File(path + "/bad_chs.mat", 'r')['bad_chs']
                    bad_chan = matfile[()].T
                except IOError:
                    bad_chan = scipy.io.loadmat(path + "/bad_chs.mat")['bad_chs']
                if len(bad_chan) == 0:
                    bad_chan = np.zeros((len(self.labels), 1))
            else:
                bad_chan = np.zeros((len(self.labels), 1))
            try:
                badchans = pd.read_csv(path_save + '\\' + folder + '/data/badchan.csv')
                badchans = badchans.drop(columns=str(p), errors='ignore')
                badchans.insert(loc=1, column=str(p), value=bad_chan[:, 0])
                # new_column = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                # badchans[str(block)] = bad_chan[:, 0]
            except FileNotFoundError:
                badchans = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(p): bad_chan[:, 0]})
            badchans.to_csv(path_save + '\\' + folder + '/data/badchan.csv', index=False,
                            header=True)  # scat_plot

            # get data
            EEG_block = np.zeros((len(self.labels), len(stim_table), self.dur_tot * self.Fs))
            EEG_block[:, :, :] = np.NaN
            # load matlab EEG
            try:
                matfile = h5py.File(path + "/ppEEG.mat", 'r')['ppEEG']
                EEGpp = matfile[()].T
            except IOError:
                EEGpp = scipy.io.loadmat(path + "/ppEEG.mat")['ppEEG']

            print('ppEEG loaded ')
            # go through each stim trigger
            for s in range(len(stim_table)):
                trig = stim_table.TTL_DS.values[s]
                if not np.isnan(trig):
                    if np.int64(trig + self.dur[0, 1] * self.Fs) > EEGpp.shape[1]:
                        EEG_block[:, s, 0:EEGpp.shape[1] - np.int64(trig + self.dur[0, 0] * self.Fs)] = EEGpp[:,
                                                                                                        np.int64(
                                                                                                            trig +
                                                                                                            self.dur[
                                                                                                                0, 0] * self.Fs):
                                                                                                        EEGpp.shape[
                                                                                                            1]]
                    elif np.int64(trig + self.dur[0, 0] * self.Fs) < 0:
                        EEG_block[:, s, abs(np.int64(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:, 0:np.int64(
                            trig + self.dur[0, 1] * self.Fs)]
                    else:
                        EEG_block[:, s, :] = EEGpp[:, np.int64(trig + self.dur[0, 0] * self.Fs):np.int64(
                            trig + self.dur[0, 1] * self.Fs)]

            np.save(file_block,
                    EEG_block)
            stim_table.to_csv(path_save + '\\' + folder + '/data/Stim_list_' + str(p).zfill(
                2) + '_' + t + '.csv', index=False,
                              header=True)  # scat_plot

    def cut_resp_IOM(self, path_pp, path_save, prot='IOM'):
        # get data
        num_stim = 20 * 60  # number of stimulation (20minutes) # hardcoded !
        # load matlab EEG
        try:
            matfile = h5py.File(path_pp + "/ppEEG.mat", 'r')['ppEEG']
            EEGpp = matfile[()].T
        except IOError:
            EEGpp = scipy.io.loadmat(path_pp + "/ppEEG.mat")['ppEEG']
        trig = np.arange(0, self.Fs * (num_stim + 1), 500).astype('int')  # all trigger. every second
        EEG_block = np.zeros((len(self.labels), len(trig), self.dur_tot * self.Fs))
        EEG_block[:, :, :] = np.NaN
        #
        for t, s in zip(trig, np.arange(num_stim)):
            EEG_block[:, s, :] = EEGpp[:,
                                 np.int64(trig + self.dur[0, 0] * self.Fs):np.int64(trig + self.dur[0, 1] * self.Fs)]

        # todo:
        np.save(path_save + '\\Epoch_data_' + prot + '.npy', EEG_block)

    def cut_resp(self, path, block, type, skip = 1):
        ###MAIN FUNCTION
        # infos, always the same
        if type == 'PP':
            types = ['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP']
            folder = 'PairedPulse'
        elif type == 'IO':
            types = ['IO', 'Ph_IO', 'CR_IO', 'InputOutput']
            folder = 'InputOutput'
        elif type == 'BM':
            types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
            folder = 'BrainMapping'
        elif type == 'LT':
            types = ['LTD1', 'LTD10', 'LTP50']
            folder = 'LongTermInduction'
        else:
            types = []
            folder = 'nofolder'
        if len(types) > 0:
            # Patient specific
            filename = ntpath.basename(path)
            data_path = os.path.dirname(os.path.dirname(path))
            subj = filename[0:5]  # EL000
            # data
            path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))

            t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
            if filename[-1].isnumeric():
                if filename[-2].isnumeric():
                    t = filename[9:-2]
                    p = int(filename[-2:])
                else:
                    t = filename[9:-1]
                    p = int(filename[-1])
                if t[-1] == '_':
                    t = t[:-1]
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                           sheet_name='Sheet' + str(p))  #
            else:
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
                p = 0
            condition = t[0:2]
            filename_block = self.path_patient_analysis + '\\' + folder + '/data/All_resps_' + str(block).zfill(
                    2) + '_' + condition + str(p).zfill(2) + '.npy'
            short_name = str(block).zfill(
                    2) + '_' + condition + str(p).zfill(2)
            if os.path.isfile(filename_block)*skip:
                print(short_name+' already exist', end='\r')
            else:
                print(short_name + ' cutting', end='\r')
                if type == 'LT':
                    stim_table = stim_table[stim_table['type'].str.contains('|'.join(types))]
                    # remove induction stimulations from stim list. we dont anaylze these for now..
                    stim_table = stim_table[~stim_table['type'].str.contains('Prot')]
                    folder = folder + '\\' + t
                else:
                    stim_table = stim_table[stim_table['type'].isin(types)]
                if not os.path.exists(self.path_patient_analysis + '\\' + folder):
                    os.makedirs(self.path_patient_analysis + '\\' + folder)
                    os.makedirs(self.path_patient_analysis + '\\' + folder + '\\data')

                stim_table = stim_table.drop(columns="Num", errors='ignore')
                stim_table = stim_table.reset_index(drop=True)
                stim_table = stim_table[stim_table.ChanP > 0]
                stim_table = stim_table[stim_table.ChanN > 0]
                stim_table = stim_table.reset_index(drop=True)
                stim_table.insert(10, "Num", np.arange(0, len(stim_table), True))
                if len(stim_table) > 0:
                    if not os.path.exists(self.path_patient_analysis + '\\' + folder + '/data/'):
                        os.makedirs(self.path_patient_analysis + '\\' + folder + '/data/')
                    # Get bad channels
                    if os.path.isfile(path + "/bad_chs.mat"):
                        try:  # load bad channels
                            matfile = h5py.File(path + "/bad_chs.mat", 'r')['bad_chs']
                            bad_chan = matfile[()].T
                        except IOError:
                            bad_chan = scipy.io.loadmat(path + "/bad_chs.mat")['bad_chs']
                        if len(bad_chan) == 0:
                            bad_chan = np.zeros((len(self.labels), 1))
                    else:
                        bad_chan = np.zeros((len(self.labels), 1))
                    try:
                        badchans = pd.read_csv(self.path_patient_analysis + '\\' + folder + '/data/badchan.csv')
                        badchans = badchans.drop(columns=str(block), errors='ignore')
                        badchans.insert(loc=1, column=str(block), value=bad_chan[:, 0])
                        # new_column = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                        # badchans[str(block)] = bad_chan[:, 0]
                    except FileNotFoundError:
                        badchans = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                    badchans.to_csv(self.path_patient_analysis + '\\' + folder + '/data/badchan.csv', index=False,
                                    header=True)  # scat_plot

                    # get data
                    EEG_block = np.zeros((len(self.labels), len(stim_table), self.dur_tot * self.Fs))
                    EEG_block[:, :, :] = np.NaN
                    # load matlab EEG
                    try:
                        matfile = h5py.File(path + "/ppEEG.mat", 'r')['ppEEG']
                        EEGpp = matfile[()].T
                    except IOError:
                        EEGpp = scipy.io.loadmat(path + "/ppEEG.mat")['ppEEG']

                    print('ppEEG loaded ')
                    # go through each stim trigger
                    for s in range(len(stim_table)):
                        trig = stim_table.TTL_DS.values[s]
                        if not np.isnan(trig):
                            if np.int64(trig + self.dur[0, 1] * self.Fs) > EEGpp.shape[1]:
                                EEG_block[:, s, 0:EEGpp.shape[1] - np.int64(trig + self.dur[0, 0] * self.Fs)] = EEGpp[:,
                                                                                                                np.int64(
                                                                                                                    trig +
                                                                                                                    self.dur[
                                                                                                                        0, 0] * self.Fs):
                                                                                                                EEGpp.shape[
                                                                                                                    1]]
                            elif np.int64(trig + self.dur[0, 0] * self.Fs) < 0:
                                EEG_block[:, s, abs(np.int64(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:, 0:np.int64(
                                    trig + self.dur[0, 1] * self.Fs)]
                            else:
                                EEG_block[:, s, :] = EEGpp[:, np.int64(trig + self.dur[0, 0] * self.Fs):np.int64(
                                    trig + self.dur[0, 1] * self.Fs)]

                    np.save(filename_block,
                            EEG_block)
                    stim_table.to_csv(self.path_patient_analysis + '\\' + folder + '/data/Stim_list_' + str(block).zfill(
                        2) + '_' + condition + str(p).zfill(2) + '.csv', index=False,
                                      header=True)  # scat_plot
                else:
                    print('No Stimulation in this protocol')
        else:
            print('ERROR: no type defined (BM, IO, PP)')

    def cut_osc(self, path, block, type):
        # infos, always the same
        if type == 'PP':
            types = ['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP']
            folder = 'PairedPulse'
        elif type == 'IO':
            types = ['IO', 'Ph_IO', 'CR_IO', 'InputOutput']
            folder = 'InputOutput'
        elif type == 'BM':
            types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
            folder = 'BrainMapping'
        else:
            types = []
            folder = 'nofolder'
        if len(types) > 0:
            # Patient specific
            filename = ntpath.basename(path)
            data_path = os.path.dirname(os.path.dirname(path))
            subj = filename[0:5]  # EL000
            # data
            path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))

            t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
            if filename[-1].isnumeric():
                if filename[-2].isnumeric():
                    t = filename[9:-2]
                    p = int(filename[-2:])
                else:
                    t = filename[9:-1]
                    p = int(filename[-1])
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                           sheet_name='Sheet' + str(p))  #
            else:
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
            print(t)
            condition = t[0:2]
            stim_table = stim_table[stim_table['type'].isin(types)]

            stim_table = stim_table.drop(columns="Num", errors='ignore')
            stim_table.insert(10, "Num", np.arange(0, len(stim_table), True))
            if len(stim_table) > 0:
                ##HGP
                HGP = np.load(path + '/HGP.npy')
                HG_block = np.zeros((len(HGP), len(self.labels), len(stim_table), self.dur_tot * self.Fs))
                SOP = np.load(path + '/HGP.npy')
                SO_block = np.zeros((len(SOP), len(self.labels), len(stim_table), self.dur_tot * self.Fs))
                print('HGP loaded ')
                for s in range(len(stim_table)):
                    trig = stim_table.TTL_DS.values[s]
                    HG_block[:, :, s, :] = HGP[:, :, np.int64(trig + self.dur[0, 0] * self.Fs):np.int64(
                        trig + self.dur[0, 1] * self.Fs)]
                    SO_block[:, :, s, :] = SOP[:, :, np.int64(trig + self.dur[0, 0] * self.Fs):np.int64(
                        trig + self.dur[0, 1] * self.Fs)]
                np.save(
                    path_patient + '/Analysis/' + folder + '/data/HGP_resps_' + str(block) + '_' + condition + '.npy',
                    HG_block)
                np.save(
                    path_patient + '/Analysis/' + folder + '/data/SOP_resps_' + str(block) + '_' + condition + '.npy',
                    SO_block)
            else:
                print('No Stimulation in this protocol')
        else:
            print('ERROR: no type defined (BM, IO, PP)')

    def list_update(self, path, block, type):
        # infos, always the same
        if type == 'PP':
            types = ['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP']
            folder = 'PairedPulse'
        elif type == 'IO':
            types = ['IO', 'Ph_IO', 'CR_IO', 'InputOutput']
            folder = 'InputOutput'
        elif type == 'BM':
            types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
            folder = 'BrainMapping'
        else:
            types = []
            folder = 'nofolder'
        if len(types) > 0:
            # Patient specific
            filename = ntpath.basename(path)
            data_path = os.path.dirname(os.path.dirname(path))
            subj = filename[0:5]  # EL000
            # data
            path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))
            t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-

            if filename[-1].isnumeric():
                if filename[-2].isnumeric():
                    t = filename[9:-2]
                    p = int(filename[-2:])
                else:
                    t = filename[9:-1]
                    p = int(filename[-1])
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                           sheet_name='Sheet' + str(p))  #
            else:
                stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
                p = 0
            print(t)
            condition = t[0:2]
            if type == 'LT':
                stim_table = stim_table[stim_table['type'].str.contains('|'.join(types))]
            else:
                stim_table = stim_table[stim_table['type'].isin(types)]

            stim_table = stim_table.drop(columns="Num", errors='ignore')
            stim_table.insert(10, "Num", np.arange(0, len(stim_table), True))
            if len(stim_table) > 0:
                # Get bad channels
                if os.path.isfile(path + "/bad_chs.mat"):
                    try:  # load bad channels
                        matfile = h5py.File(path + "/bad_chs.mat", 'r')['bad_chs']
                        bad_chan = matfile[()].T
                    except IOError:
                        bad_chan = scipy.io.loadmat(path + "/bad_chs.mat")['bad_chs']
                    if len(bad_chan) == 0:
                        bad_chan = np.zeros((len(self.labels), 1))
                else:
                    bad_chan = np.zeros((len(self.labels), 1))
                try:
                    badchans = pd.read_csv(self.path_patient_analysis + '\\' + folder + '/data/badchan.csv')
                    badchans = badchans.drop(columns=str(block), errors='ignore')
                    badchans.insert(loc=1, column=str(block), value=bad_chan[:, 0])
                    # new_column = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                    # badchans[str(block)] = bad_chan[:, 0]
                except FileNotFoundError:
                    badchans = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
                badchans.to_csv(self.path_patient_analysis + '\\' + folder + '/data/badchan.csv', index=False,
                                header=True)  # scat_plot
                # todo: two digit number of block
                col_drop = ["StimNum", 'StimNum.1', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS',
                            'currentflow']
                for d in range(len(col_drop)):
                    if (col_drop[d] in stim_table.columns):
                        stim_table = stim_table.drop(columns=col_drop[d])
                stim_table.insert(0, "StimNum", np.arange(len(stim_table)), True)
                stim_table = stim_table.reset_index(drop=True)

                stim_table.to_csv(self.path_patient_analysis + '\\' + folder + '/data/Stim_list_' + str(block).zfill(
                    2) + '_' + condition + str(p).zfill(2) + '.csv', index=False,
                                  header=True)  # scat_plot

                print('stimlist updated')
        else:
            print('ERROR: no valid type name')

    def concat_resp(self, type):
        if type == 'PP':
            types = ['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP']
            folder = 'PairedPulse'
        elif type == 'IO':
            types = ['IO', 'Ph_IO', 'CR_IO', 'InputOutput']
            folder = 'InputOutput'
        elif type == 'BM':
            types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
            folder = 'BrainMapping'
        else:
            types = []
            folder = 'nofolder'

        files = glob(self.path_patient + '/Analysis/' + folder + '/data/Stim_list_*')
        files = np.sort(files)
        # prots           = np.int64(np.arange(1, len(files) + 1))  # 43
        stimlist = []
        EEG_resp = []
        conds = np.empty((len(files),), dtype=object)
        for p in range(len(files)):
            file = files[p]
            # file = glob(self.path_patient + '/Analysis/'+folder+'/data/Stim_list_' + str(p) + '_*')[0]
            idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
            cond = ntpath.basename(file)[idxs[-2] - 2:idxs[-2]]  # ntpath.basename(file)[idxs[-2] + 2:-4]  #
            conds[p] = cond
            k = int(ntpath.basename(file)[idxs[0]:idxs[1] + 1])
            EEG_block = np.load(self.path_patient + '/Analysis/' + folder + '/data/All_resps_' + file[-11:-4] + '.npy')
            print(str(p + 1) + '/' + str(len(files)) + ' -- All_resps_' + file[-11:-4])
            stim_table = pd.read_csv(file)
            stim_table['type'] = cond
            if len(stimlist) == 0:
                EEG_resp = EEG_block
                stimlist = stim_table
            else:
                EEG_resp = np.concatenate([EEG_resp, EEG_block], axis=1)
                stimlist = pd.concat([stimlist, stim_table])
            # del EEG_block
            # os.remove(self.path_patient + '/Analysis/'+folder+'/data/All_resps_' + str(k) + '_' + cond + '.npy')
            # os.remove(self.path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(p) + '_' + cond + '.csv')
        stimlist = stimlist.drop(columns="StimNum", errors='ignore')

        # stimlist.loc[(stimlist.Condition == 'Benzo'), 'sleep']  = 5
        # stimlist.loc[(stimlist.Condition == 'Fuma'), 'sleep']   = 6
        stimlist = stimlist.fillna(0)
        stimlist = stimlist.reset_index(drop=True)
        col_drop = ["StimNum", 'StimNum.1', 's', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
        for d in range(len(col_drop)):
            if (col_drop[d] in stimlist.columns):
                stimlist = stimlist.drop(columns=col_drop[d])
        stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)

        np.save(self.path_patient + '/Analysis/' + folder + '/data/All_resps.npy', EEG_resp)
        stimlist.to_csv(self.path_patient + '/Analysis/' + folder + '/data/Stimlist.csv', index=False,
                        header=True)  # scat_plot
        print('data stored')
        print(self.path_patient + '/Analysis/' + folder + '/data/All_resps.npy')

    def concat_osc(self, type):
        if type == 'PP':
            types = ['CR', 'PP', 'Ph_PP', 'CR_PP', 'Circadian PP']
            folder = 'PairedPulse'
        elif type == 'IO':
            types = ['IO', 'Ph_IO', 'CR_IO', 'InputOutput']
            folder = 'InputOutput'
        elif type == 'BM':
            types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
            folder = 'BrainMapping'
        else:
            types = []
            folder = 'nofolder'

        files = glob(self.path_patient + '/Analysis/' + folder + '/data/Stim_list_*')
        files = np.sort(files)
        # prots           = np.int64(np.arange(1, len(files) + 1))  # 43
        stimlist = []
        HGP_resp = []
        SOP_resp = []
        conds = np.empty((len(files),), dtype=object)
        for p in range(len(files)):
            file = files[p]  # stimlist
            idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
            cond = ntpath.basename(file)[idxs[-1] + 2:-4]  #
            conds[p] = cond
            block = int(ntpath.basename(file)[idxs[0]:idxs[-1] + 1])
            HGP_block = np.load(
                self.path_patient + '/Analysis/' + folder + '/data/HGP_resps_' + str(block) + '_' + cond + '.npy')
            SOP_block = np.load(
                self.path_patient + '/Analysis/' + folder + '/data/HGP_resps_' + str(block) + '_' + cond + '.npy')
            if p == 0:
                HGP_resp = HGP_block
                SOP_resp = SOP_block

            else:
                HGP_resp = np.concatenate([HGP_resp, HGP_block], axis=2)
                SOP_resp = np.concatenate([SOP_resp, SOP_block], axis=2)

            ## z-score
        HGP_z = np.zeros((len(self.labels_all), HGP_resp.shape[2], HGP_resp.shape[3]))
        for f in range(len(HGP_resp)):
            mean_b = np.nanmean(HGP_resp[f, :, :, np.int(0.7 * self.Fs):np.int(0.98 * self.Fs)], axis=(1, 2))
            std_b = np.nanstd(HGP_resp[f, :, :, np.int(0.7 * self.Fs):np.int(0.98 * self.Fs)], axis=(1, 2))
            for c in range(len(self.labels_all)):
                HGP_z[c] = HGP_z[c] + (HGP_resp[f, c] - mean_b[c]) / std_b[c]
        HGP_z = HGP_z / len(HGP_resp)
        np.save(self.path_patient + '/Analysis/BrainMapping/data/All_HGPz.npy', HGP_z)
        np.save(self.path_patient + '/Analysis/BrainMapping/data/All_SOP.npy', SOP_resp)

    def concat_list(self, type):
        if type == 'PP':
            folder = 'PairedPulse'
        elif type == 'IO':
            folder = 'InputOutput'
        elif type == 'BM':
            folder = 'BrainMapping'
        else:
            folder = 'nofolder'

        files = glob(self.path_patient_analysis + '\\' + folder + '/data/Stim_list_*')
        files = np.sort(files)
        # prots           = np.int64(np.arange(1, len(files) + 1))  # 43
        stimlist = []
        conds = np.empty((len(files),), dtype=object)
        for p in range(len(files)):
            file = files[p]
            # file = glob(self.path_patient + '/Analysis/'+folder+'/data/Stim_list_' + str(p) + '_*')[0]
            idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]

            cond = ntpath.basename(file)[idxs[-2] - 2:idxs[-2]]  # ntpath.basename(file)[idxs[-2] + 2:-4]  #
            conds[p] = cond
            k = int(ntpath.basename(file)[idxs[0]:idxs[1] + 1])
            stim_table = pd.read_csv(file)
            stim_table['type'] = cond
            if len(stimlist) == 0:
                stimlist = stim_table
            else:
                stimlist = pd.concat([stimlist, stim_table])
            # os.remove(self.path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(p) + '_' + cond + '.csv')

        col_drop = ['StimNum', 'StimNum.1', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
        for d in range(len(col_drop)):
            if (col_drop[d] in stimlist.columns):
                stimlist = stimlist.drop(columns=col_drop[d])
        stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)

        stimlist.to_csv(self.path_patient_analysis + '\\' + folder + '/data/Stimlist.csv', index=False,
                        header=True)  # scat_plot
        print('data stored')
        print(self.path_patient + '/Analysis/' + folder + '/data/Stimlist.csv')

    def cut_BM(self, path, block):

        # infos, always the same
        types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
        # Patient specific
        filename = ntpath.basename(path)
        data_path = os.path.dirname(os.path.dirname(path))
        subj = filename[0:5]  # EL000
        # data
        path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))

        t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
        if filename[-1].isnumeric():
            if filename[-2].isnumeric():
                t = filename[9:-2]
                p = int(filename[-2:])
            else:
                t = filename[9:-1]
                p = int(filename[-1])
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                       sheet_name='Sheet' + str(p))  #
        else:
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
        print(t)
        condition = t[0:2]
        stim_table = stim_table[stim_table['type'].isin(types)]
        # get data
        EEG_block = np.zeros((len(self.labels), len(stim_table), self.dur_tot * self.Fs))
        EEG_block[:, :, :] = np.NaN
        try:
            matfile = h5py.File(path + "/ppEEG.mat", 'r')['ppEEG']
            EEGpp = matfile[()].T
        except IOError:
            EEGpp = scipy.io.loadmat(path + "/ppEEG.mat")['ppEEG']

        print('ppEEG loaded ')
        for s in range(len(stim_table)):
            trig = stim_table.TTL_DS.values[s]
            if np.int(trig + self.dur[0, 1] * self.Fs) > EEGpp.shape[1]:
                EEG_block[:, s, 0:EEGpp.shape[1] - np.int(trig + self.dur[0, 0] * self.Fs)] = EEGpp[:,
                                                                                              np.int(trig + self.dur[
                                                                                                  0, 0] * self.Fs):
                                                                                              EEGpp.shape[1]]
            elif np.int(trig + self.dur[0, 0] * self.Fs) < 0:
                EEG_block[:, s, abs(np.int(trig + self.dur[0, 0] * self.Fs)):] = EEGpp[:, 0:np.int(
                    trig + self.dur[0, 1] * self.Fs)]
            else:
                EEG_block[:, s, :] = EEGpp[:,
                                     np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]

        np.save(path_patient + '/Analysis/BrainMapping/data/All_resps_' + str(block) + '_' + condition + '.npy',
                EEG_block)
        stim_table.to_csv(
            path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(block) + '_' + condition + '.csv',
            index=False,
            header=True)  # scat_plot

    def cut_BM_HGP(self, path, block):

        # infos, always the same
        types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
        # Patient specific
        filename = ntpath.basename(path)
        data_path = os.path.dirname(os.path.dirname(path))
        subj = filename[0:5]  # EL000
        # data
        path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))

        t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
        if filename[-1].isnumeric():
            if filename[-2].isnumeric():
                t = filename[9:-2]
                p = int(filename[-2:])
            else:
                t = filename[9:-1]
                p = int(filename[-1])
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                       sheet_name='Sheet' + str(p))  #
        else:
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
        print(t)
        condition = t[0:2]
        stim_table = stim_table[stim_table['type'].isin(types)]
        if os.path.isfile(path + "/bad_chs.mat"):
            try:  # load bad channels
                matfile = h5py.File(path + "/bad_chs.mat", 'r')['bad_chs']
                bad_chan = matfile[()].T
            except IOError:
                bad_chan = scipy.io.loadmat(path + "/bad_chs.mat")['bad_chs']
        else:
            bad_chan = np.zeros((len(self.labels), 1))
        try:
            badchans = pd.read_csv(path_patient + '/Analysis/BrainMapping/data/badchan.csv')
            badchans[str(block)] = bad_chan[:, 0]
        except FileNotFoundError:
            badchans = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
        badchans.to_csv(path_patient + '/Analysis/BrainMapping/data/badchan.csv', index=False,
                        header=True)  # scat_plot
        if len(stim_table) > 0:
            # get data

            HGP = np.load(path + '/HGP.npy')
            HG_block = np.zeros((len(HGP), len(self.labels), len(stim_table), self.dur_tot * self.Fs))
            print('HGP loaded ')
            for s in range(len(stim_table)):
                trig = stim_table.TTL_DS.values[s]
                HG_block[:, :, s, :] = HGP[:, :,
                                       np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]

            np.save(path_patient + '/Analysis/BrainMapping/data/HGP_resps_' + str(block) + '_' + condition + '.npy',
                    HG_block)

    def cut_BM_SOP(self, path, block):

        # infos, always the same
        types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
        # Patient specific
        filename = ntpath.basename(path)
        data_path = os.path.dirname(os.path.dirname(path))
        subj = filename[0:5]  # EL000
        # data
        path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))

        t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-
        if filename[-1].isnumeric():
            if filename[-2].isnumeric():
                t = filename[9:-2]
                p = int(filename[-2:])
            else:
                t = filename[9:-1]
                p = int(filename[-1])
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                       sheet_name='Sheet' + str(p))  #
        else:
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
        print(t)
        condition = t[0:2]
        stim_table = stim_table[stim_table['type'].isin(types)]
        if len(stim_table) > 0:
            # get data

            HGP = np.load(path + '/SOP.npy')
            HG_block = np.zeros((len(HGP), len(self.labels), len(stim_table), self.dur_tot * self.Fs))
            print('HGP loaded ')
            for s in range(len(stim_table)):
                trig = stim_table.TTL_DS.values[s]
                HG_block[:, :, s, :] = HGP[:, :,
                                       np.int(trig + self.dur[0, 0] * self.Fs):np.int(trig + self.dur[0, 1] * self.Fs)]

            np.save(path_patient + '/Analysis/BrainMapping/data/SOP_resps_' + str(block) + '_' + condition + '.npy',
                    HG_block)

    def cut_BM_updatelist(self, path, block):
        # infos, always the same
        types = ['BM', 'Ph_BM', 'CR_BM', 'BrainMapping']
        # Patient specific
        filename = ntpath.basename(path)
        data_path = os.path.dirname(os.path.dirname(path))
        subj = filename[0:5]  # EL000
        # data
        path_patient = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        t = filename[9:]  # BM, CR_BM, Ph_BM, Ph etc-

        if filename[-1].isnumeric():
            if filename[-2].isnumeric():
                t = filename[9:-2]
                p = int(filename[-2:])
            else:
                t = filename[9:-1]
                p = int(filename[-1])
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx",
                                       sheet_name='Sheet' + str(p))  #
        else:
            stim_table = pd.read_excel(data_path + "/" + subj + "_stimlist_" + t + ".xlsx")  #
        print(t)
        condition = t[0:2]
        stim_table = stim_table[stim_table['type'].isin(types)]
        if os.path.isfile(path + "/bad_chs.mat"):
            try:  # load bad channels
                matfile = h5py.File(path + "/bad_chs.mat", 'r')['bad_chs']
                bad_chan = matfile[()].T
            except IOError:
                bad_chan = scipy.io.loadmat(path + "/bad_chs.mat")['bad_chs']
        else:
            bad_chan = np.zeros((len(self.labels), 1))
        try:
            badchans = pd.read_csv(path_patient + '/Analysis/BrainMapping/data/badchan.csv')
            badchans[str(block)] = bad_chan[:, 0]
        except FileNotFoundError:
            badchans = pd.DataFrame({'Chan': np.arange(len(bad_chan)), str(block): bad_chan[:, 0]})
        badchans.to_csv(path_patient + '/Analysis/BrainMapping/data/badchan.csv', index=False,
                        header=True)  # scat_plot

        stim_table.to_csv(
            path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(block) + '_' + condition + '.csv',
            index=False,
            header=True)  # scat_plot
        print('stimlist updated')

    def concat_BM(self, EEGup=False):
        files = glob(self.path_patient + '/Analysis/BrainMapping/data/Stim_list_*')
        prots = np.int64(np.arange(1, len(files) + 1))  # 43
        EEG_resp = []
        stimlist = []
        conds = np.empty((prots.shape), dtype=object)
        for p in prots:
            print(p)
            file = glob(self.path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(p) + '_*')[0]
            idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
            cond = ntpath.basename(file)[idxs[-1] + 2:-4]  #
            conds[p - 1] = cond
            if EEGup:
                EEG_block = np.load(
                    self.path_patient + '/Analysis/BrainMapping/data/All_resps_' + str(p) + '_' + cond + '.npy')
            stim_table = pd.read_csv(
                self.path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(p) + '_' + cond + '.csv')
            stim_table['Condition'] = cond
            if len(stimlist) == 0:
                if EEGup:
                    EEG_resp = EEG_block
                stimlist = stim_table
            else:
                if EEGup:
                    EEG_resp = np.concatenate([EEG_resp, EEG_block], axis=1)
                stimlist = pd.concat([stimlist, stim_table])
            # del EEG_block
            if EEGup:
                os.remove(self.path_patient + '/Analysis/BrainMapping/data/All_resps_' + str(p) + '_' + cond + '.npy')
            # os.remove(self.path_patient + '/Analysis/BrainMapping/data/Stim_list_' + str(p) + '_' + cond + '.csv')
        stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)
        stimlist.loc[(stimlist.Condition == 'Benzo'), 'sleep'] = 5
        stimlist.loc[(stimlist.Condition == 'Fuma'), 'sleep'] = 6
        stimlist = stimlist.fillna(0)
        if EEGup:
            np.save(self.path_patient + '/Analysis/BrainMapping/data/All_resps.npy', EEG_resp)
        stimlist.to_csv(self.path_patient + '/Analysis/BrainMapping/data/Stimlist.csv', index=False,
                        header=True)  # scat_plot
        print('data stored')
        print(self.path_patient + '/Analysis/BrainMapping/data/All_resps.npy')

    def concat_BM_HGP(self):
        files = glob(self.path_patient + '/Analysis/BrainMapping/data/HGP_resps_*')
        HGP_resp = []
        conds = np.empty((len(files), 1), dtype=object)
        for p in range(len(files)):
            file = files[p]
            idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
            cond = ntpath.basename(file)[idxs[-1] + 2:-4]  #
            conds[p] = cond
            HGP_block = np.load(file)

            if p == 0:
                HGP_resp = HGP_block

            else:
                HGP_resp = np.concatenate([HGP_resp, HGP_block], axis=2)
            # del EEG_block
            # os.remove(file)
            #### z-score
        ## z-score
        HGP_z = np.zeros((len(self.labels_all), HGP_resp.shape[2], HGP_resp.shape[3]))
        for f in range(len(HGP_resp)):
            mean_b = np.nanmean(HGP_resp[f, :, :, np.int(0.7 * self.Fs):np.int(0.98 * self.Fs)], axis=(1, 2))
            std_b = np.nanstd(HGP_resp[f, :, :, np.int(0.7 * self.Fs):np.int(0.98 * self.Fs)], axis=(1, 2))
            for c in range(len(self.labels_all)):
                HGP_z[c] = HGP_z[c] + (HGP_resp[f, c] - mean_b[c]) / std_b[c]

        HGP_z = HGP_z / len(HGP_resp)

        np.save(self.path_patient + '/Analysis/BrainMapping/data/All_HGPz.npy', HGP_z)
        print('data stored')
        print(self.path_patient + '/Analysis/BrainMapping/data/All_HGPz.npy')

    def concat_BM_SOP(self):
        files = glob(self.path_patient + '/Analysis/BrainMapping/data/SOP_resps_*')
        SOP_resp = []
        conds = np.empty((len(files), 1), dtype=object)
        for p in range(len(files)):
            file = files[p]
            idxs = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
            cond = ntpath.basename(file)[idxs[-1] + 2:-4]  #
            conds[p] = cond
            HGP_block = np.load(file)

            if p == 0:
                SOP_resp = HGP_block

            else:
                SOP_resp = np.concatenate([SOP_resp, HGP_block], axis=2)
            # del EEG_block
            # os.remove(file)
            #### z-score
        # ## z-score
        # HGP_z = np.zeros((len(self.labels_all), HGP_resp.shape[2], HGP_resp.shape[3]))
        # for f in range(len(HGP_resp)):
        #     mean_b      = np.nanmean(HGP_resp[f, :, :, np.int(0.7 * Fs):np.int(0.98 * Fs)], axis=(1, 2))
        #     std_b       = np.nanstd(HGP_resp[f, :, :, np.int(0.7 * Fs):np.int(0.98 * Fs)], axis=(1, 2))
        #     for c in range(len(self.labels_all)):
        #         HGP_z[c] = HGP_z[c] + (HGP_resp[f, c] - mean_b[c]) / std_b[c]
        #
        # HGP_z = HGP_z / len(HGP_resp)

        np.save(self.path_patient + '/Analysis/BrainMapping/data/All_SOP.npy', SOP_resp)
        print('data stored')
        print(self.path_patient + '/Analysis/BrainMapping/data/All_SOP.npy')
