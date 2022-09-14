import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib
import os
import dabest

subj            = "EL003"
cwd             = os.getcwd()
path_patient    = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))+'/Patients/'+subj
Int_all         = [1,2,4]
StimChan ='HippAnt10'
w=0.25
cs =[0,1,2,3]
LL_all  = pd.read_csv(path_patient + '/Analysis/Pharmacology/LL/LL_CR_' + StimChan + '_' + str(w) + 's.csv')

for Int in Int_all:
    data_LL = LL_all[(LL_all['Int'] == Int) & (LL_all['IPI'] > 1000 * w) & (LL_all['Chan'].isin(cs))]
    data = pd.DataFrame({'Baseline': data_LL[data_LL.State == 0]['LL SP norm BL'].values[0:36],
                         'Benzodiazepine': data_LL[data_LL.State == 2]['LL SP norm BL'].values[0:36]})
    two_groups_unpaired = dabest.load(data, idx=("Baseline", "Benzodiazepine"), ci=95, resamples=5000)
    two_groups_unpaired.mean_diff.plot()
