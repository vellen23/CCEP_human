import sys
sys.path.append('./funcs/')
import pandas as pd
import read_data as rd
import plot_funcs as pf
import plot_connectogram
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
sub_path  ='X:\\4 e-Lab\\' # y:\\eLab



## 1. get data
data_con_file = sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\data_con_all.csv'
data_con_all = pd.read_csv(data_con_file)
data_con = data_con_all[~np.isnan(data_con_all.Dir_index)]
data_con = data_con.reset_index(drop=True)
chan_ID = np.unique(np.concatenate([data_con.Stim, data_con.Chan])).astype('int')
data_nodes = rd.get_nodes(chan_ID, data_con)
#
### Load plot information
plot_main = plot_connectogram.main_plot(data_con, data_nodes)

# edges that will be modified
data_edges = data_con[(data_con.d > 80) & (data_con.d < 100) & (data_con.Dir_index == 1)]
data_edges = data_edges.reset_index(drop=True)

##figure

plot_main.plot_con(data_edges,0)
print('plot done')