import os
import numpy as np
import imageio

import sys
sys.path.append('./py_functions')
from glob import glob
subj            = "EL011"
cwd             = os.getcwd()
def create(path, name):
    ## GIF
    filenames = glob(path + '/*.jpg')
    print('Creating charts\n')

    with imageio.get_writer(path + '/' + name + '.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')
    print('Removing Images\n')
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

for subj in ['EL010', 'EL011', 'EL014', 'EL015']:
#path = '/Volumes/EvM_T7/PhD/EL_experiment/Patients/EL014/Analysis/BrainMapping/CR/figures/BM_LL/GIF/'
    path = 'T:\EL_experiment\Patients\\'+subj+'\Analysis\BrainMapping\\CR\\figures\BM_prob\GIF'#'T:\EL_experiment\Patients\\'+subj+'\Analysis\BrainMapping\LL\\figures\BM_plot\GIF'
    name = 'BM_CR'  # 'BM_Ph'

    create(path, name)