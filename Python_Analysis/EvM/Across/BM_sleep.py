import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd

import sys

sys.path.append('T:\EL_experiment\Codes\CCEP_human\Python_Analysis/py_functions')

import load_summary as ls
import BM_across_plots as plotting

cwd = os.getcwd()

##all
cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]
dist_groups = np.array([[0, 15], [15, 30], [30, 5000]])
dist_labels = ['local (<15 mm)', 'short (<30mm)', 'long']
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab
# General data
regions_all = pd.read_excel("X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients\\Across\\elab_labels.xlsx",sheet_name="region_sort")
regions_all = regions_all.label.values


##
subjs = ['EL010', 'EL011', 'EL014', 'EL015', 'EL016', 'EL017', 'EL019', 'EL020']
data_con_file = sub_path + '\EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\Sleep\connectogram\\data_con_stat.csv'
con_file = sub_path + 'EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\data_con.csv'
degree_file = sub_path+'\EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\Sleep\\G_deg.csv'

###load data
if not os.path.exists(data_con_file):
    ls.get_connections_sleep(subjs, sub_path, data_con_file)

# connectionwise
data_con_sleep = pd.read_csv(data_con_file)
if "Group" not in data_con_sleep:  # adding DI, onset and Group to sleep ddata
    data_con = pd.read_csv(con_file)
    data_con = data_con[['Subj', 'Stim', 'Chan', 'DI', 'onset']]
    data_con = data_con.reset_index(drop=True)
    data_con_sleep = data_con_sleep.merge(data_con, on=['Subj', 'Stim', 'Chan'], how='left')
    data_con_sleep.insert(6, 'Group', 'indirect')
    data_con_sleep.loc[(data_con_sleep.d < 20) & (data_con_sleep.onset < 0.02), 'Group'] = 'local direct'
    data_con_sleep.loc[(data_con_sleep.d > 20) & (data_con_sleep.onset < 0.02), 'Group'] = 'long direct'
    data_con_sleep.to_csv(data_con_file,
                          header=True, index=False)
    print('get onset')

#3 degrees
G_deg = pd.read_csv(degree_file)
# Plot
for kind in ['strip', 'box']:
    for metric, label in zip(['LLs_n', 'Sig_n'],['strength (LL)', 'probability']):
        g = sns.catplot(x ='ChanR', y=metric, hue='SleepState', data= G_deg[(G_deg.ChanR != 'Unknown')&(G_deg.ChanR != 'Necrosis')&(G_deg.SleepState != 'Wake')], row = 'Deg',dodge = True,  kind=kind,height=8,aspect= 3.5, order = regions_all, palette=['#27348B','#D22B2B'])
        # sns.catplot(x ='ChanR', y=metric, hue='SleepState', data= G_deg[(G_deg.ChanR != 'Unknown')&(G_deg.ChanR != 'Necrosis')&(G_deg.SleepState != 'Wake')], row = 'Deg',dodge = True,  kind='box',height=8,aspect= 3.5, order = regions_all, palette=['#27348B','#D22B2B'], ax = g)
        ax = g.axes  # access a grid of 'axes' objects
        plt.ylim([0.3,1.6])
        ax[0,0].axhline(1, color='k', linewidth=2)
        ax[1,0].axhline(1, color='k', linewidth=2)
        plt.xticks(fontsize=20)
        plt.legend(fontsize=30)
        ax[1,0].set_yticks([0.5, 1, 1.5], [50, 100, 150])
        ax[0,0].set_yticks([0.5, 1, 1.5], [50, 100, 150])
        ax[1,0].tick_params(labelsize=20)
        ax[0,0].tick_params(labelsize=20)
        ax[0,0].set_title('In-Degree: connection '+label, fontsize=25)
        ax[1,0].set_title('Out-Degree: connection '+label, fontsize=25)
        ax[0,0].set_ylabel('Degree normalized to Wake [%]', fontsize=25)
        ax[1,0].set_ylabel('Degree normalized to Wake [%]', fontsize=25)
        plt.tight_layout()
        plt.savefig(sub_path+'\EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\Sleep\degree\\Deg_'+metric+'_'+kind+'.svg')
        plt.savefig(sub_path+'\EvM\Projects\EL_experiment\Analysis\Patients\Across\BrainMapping\Sleep\degree\\Deg_'+metric+'_'+kind+'.jpg')
        plt.show()
print('Nodes plot done')