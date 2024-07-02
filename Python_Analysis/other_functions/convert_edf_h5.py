"""
Convert from EDF to H5
"""

import re
import numpy as np
import h5py as h5tool
from datetime import datetime
from mne.io.edf.edf import RawEDF, read_raw_edf
import os
import glob

edfFile = '/Volumes/KMI_DATA/0a0fb052-6cd4-4f0b-8565-7d0b69836045.edf'
outputData = '/Volumes/KMI_DATA/0a0fb052-6cd4-4f0b-8565-7d0b69836045.h5'

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def load_edf(edfFilePath: str, encoding: str = 'utf8') -> RawEDF:
    # edf = read_raw_EDF(edfFilePath, encoding=encoding)
    import mne
    edf = mne.io.read_raw_edf(edfFilePath)
    # print(edf.ch_names)
    # print(edf.getSignalLabels())
    return edf


def write_new_hdf5(filePath: str):
    with h5tool.File(filePath, 'w') as file:
        gMeta = file.create_group('meta')
        gMeta.attrs['creation_date'] = datetime.now().timestamp()

        gReadMe = file.create_group('read_me')
        gAnnot = file.create_group('annotations')
        gTraces = file.create_group('traces')
        gTracesRaw = gTraces.create_group('raw')
        gCoords = file.create_group('coords')
        gAnalysis = file.create_group('analyses')


def write_edf_data(filePath: str, edfData: RawEDF):
    with h5tool.File(filePath, 'a') as file:

        # meta
        dateAndTime = edfData.info['meas_date']
        sfreq = edfData.info['sfreq']
        nSamples = edfData.n_times
        duration = nSamples / sfreq

        gMeta = file.require_group('meta')
        gMeta.attrs['start_date'] = f'{dateAndTime.date()}'
        gMeta.attrs['start_time'] = f'{dateAndTime.time()}'
        gMeta.attrs['start_timestamp'] = dateAndTime.timestamp()
        gMeta.attrs['duration'] = duration

        # annotations
        annot = edfData.annotations
        nAnnot = len(annot.description)
        gAnnot = file.require_group('annotations')
        dt = h5tool.special_dtype(vlen=str)
        text = np.array(annot.description, dtype=dt)
        gAnnot.create_dataset(name='text', data=text, maxshape=(nAnnot,))
        gAnnot.create_dataset(name='time', data=annot.onset, dtype='float64', maxshape=(nAnnot,), compression='gzip')

        # traces
        gTracesRaw = file.require_group('traces/raw')
        gTracesRaw.attrs['start_date'] = f'{dateAndTime.date()}'
        gTracesRaw.attrs['start_time'] = f'{dateAndTime.time()}'
        gTracesRaw.attrs['start_timestamp'] = dateAndTime.timestamp()
        gTracesRaw.attrs['n_samples'] = nSamples
        gTracesRaw.attrs['sfreq'] = sfreq
        gTracesRaw.attrs['duration'] = duration
        gTracesRaw.attrs['processing'] = 'None'

        cNames = edfData.ch_names
        for name in cNames:
            # print(f'{name} in {gTracesRaw.name}')
            if re.search('^C[0-9]+$', name) is not None \
                    and name not in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
                print(f'-- discard {name}')
                continue

            if name in ['TRIG', 'OSAT', 'PR', 'Pleth']:
                unit = 'V'
                data = edfData.get_data(picks=[name])[0]
            else:
                unit = 'uV'
                data = edfData.get_data(picks=[name], units='uV')[0]

            cleanName = re.sub('[^A-Za-z0-9_\-+]+', '', name)
            cleanName = cleanName.replace('EEG', '')
            cleanName = cleanName.replace('EOG', '')
            dset = gTracesRaw.create_dataset(name=cleanName, data=data, dtype='float32', maxshape=data.shape,
                                             chunks=(125000,), compression="gzip")
            dset.attrs['unit'] = unit
            dset.attrs['sfreq'] = sfreq


def load_annot_fromH5(h5FilePath: str):
    with h5tool.File(h5FilePath, 'r+') as h5:
        if 'traces/tmp' in h5:
            del h5['traces/tmp']

    with h5tool.File(newFile, 'w') as file:
        with h5tool.File(h5FilePath, 'r') as h5:
            for key in h5.keys():
                if 'traces' not in key:
                    file.copy(h5[key], key)

            for traceKey in h5['traces'].keys():
                if 'tmp' not in traceKey:
                    file.copy(h5['traces/' + traceKey], 'traces/' + traceKey)

        # print(h5['traces/raw'].keys())
        # text = h5['annotations/text'].asstr()[()]
        # time = h5['annotations/time'][()]
        # print('done')


def convert(edfFile, outputData, overwrite=False):
    edfData = None
    try:
        edfData = load_edf(edfFile)
    except Exception as error1:
        print(f'Cannot read EDF, will try with encoding=latin1.\nError: {error1}')
        try:
            edfData = load_edf(edfFile, encoding='latin1')
        except Exception as error2:
            print(f'Cannot read EDF.\nError: {error2}')
    if edfData is not None:
        dateAndTime = edfData.info['meas_date']
        outputData = outputData[:-3] + '_' + dateAndTime.strftime('%Y%m%d') + '_' + dateAndTime.strftime(
            '%H%M%S') + '.h5'
        if not os.path.isfile(outputData):
            write_new_hdf5(outputData)
            write_edf_data(outputData, edfData)
        else:
            if overwrite:
                write_new_hdf5(outputData)
                write_edf_data(outputData, edfData)


def convert_patient_all(subj):
    from pathlib import Path
    path_data = os.path.join(sub_path, 'Patients', subj, 'Data_Raw', 'EL_experiment')
    path_output = os.path.join(sub_path, 'Patients', subj, 'Data_Raw', 'EL_experiment', 'h5_conversion')
    Path(path_output).mkdir(
        parents=True, exist_ok=True)
    files = glob.glob(os.path.join(path_data, subj + '_*.EDF'))
    for file in files:
        # input_file = os.path.join(path_data, file)
        # output_file = os.path.join(path_output, file[:-4]+'.h5')
        _, filename = os.path.split(file)
        convert(file, os.path.join(path_output, filename[:-4] + '.h5'))


def convert_patient_file(subj, files):
    from pathlib import Path
    path_data = os.path.join(sub_path, 'Patients', subj, 'Data_Raw', 'EL_experiment')
    path_output = os.path.join(sub_path, 'Patients', subj, 'Data_Raw', 'EL_experiment', 'h5_conversion')
    Path(path_output).mkdir(
        parents=True, exist_ok=True)
    # files = glob.glob(os.path.join(path_data, subj + '_*.EDF'))
    for file in files:
        input_file = os.path.join(path_data, file)
        # output_file = os.path.join(path_output, file[:-4]+'.h5')
        _, filename = os.path.split(input_file)
        convert(input_file, os.path.join(path_output, filename[:-4] + '.h5'), overwrite=True)


if __name__ == "__main__":
    subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL017", "EL018", "EL019", "EL020", "EL021",
             "EL022", "EL024", "EL025", "EL026", "EL027"]
    subjs = ["EL027", "EL026", "EL020"]
    files = [['EL027_CR06.EDF'], ['EL026_CR04.EDF'], ['EL020_CR07.EDF', 'EL020_CR08.EDF']]

    for subj, file_sel in zip(subjs, files):
        t0 = datetime.now()
        convert_patient_file(subj, file_sel)
        print(subj, ':', datetime.now() - t0)

    print('All Subj Done')
    # load_annot_fromH5(outputData)
