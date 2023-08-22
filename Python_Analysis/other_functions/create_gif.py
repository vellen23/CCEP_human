import os
import numpy as np
import imageio
import tqdm

import sys
sys.path.append('./py_functions')
from glob import glob

def create(path, name, remove = 0):
    ## GIF
    filenames = glob(path + '/*.png')
    print('Creating charts\n')

    # imageio.imsave(exportname, frames, format='GIF', fps=30)
    with imageio.get_writer(path + '/' + name + '.gif', mode='I') as writer:
        for filename in tqdm.tqdm(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)
    print('GIF saved\n')

    # Remove files
    if remove:
        print('Removing Images\n')
        for filename in set(filenames):
            os.remove(filename)
def create_2(path, name, remove=0, dur = 1):
    # filenames = glob(path + '/*.png')
    image_folder = os.fsencode(path)
    filenames = []

    for file in os.listdir(image_folder):
        filename = os.fsdecode(file)
        if filename.endswith(('.jpeg', '.png')): #, '.gif'
            filenames.append(filename)

    filenames.sort()  # this iteration technique has no built in order, so sort the frames

    images = list(map(lambda filename: imageio.imread(os.path.join(path,filename)), filenames))

    imageio.mimsave(os.path.join(path, name + '.gif'), images, duration=dur)  # modify the frame duration as needed
    # Remove files
    if remove:
        print('Removing Images\n')
        for filename in set(filenames):
            os.remove(filename)
# for subj in ['EL010', 'EL011', 'EL014', 'EL015']:
# #path = '/Volumes/EvM_T7/PhD/EL_experiment/Patients/EL014/Analysis/BrainMapping/CR/figures/BM_LL/GIF/'
#     path = 'T:\EL_experiment\Patients\\'+subj+'\Analysis\BrainMapping\\CR\\figures\BM_prob\GIF'#'T:\EL_experiment\Patients\\'+subj+'\Analysis\BrainMapping\LL\\figures\BM_plot\GIF'
#     name = 'BM_CR'  # 'BM_Ph'
#
#     create(path, name)
path = "X:\\4 e-Lab\EvM\Projects\EPIOS\Patients\CT\step2_007"
name = 'CT_007'
create_2(path, name, dur = 0.5)