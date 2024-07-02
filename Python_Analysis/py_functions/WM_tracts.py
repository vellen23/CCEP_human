import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import load_summary as ls
import numpy as np

plt.rcParams.update({
    'font.family': 'arial',
    'font.size': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'svg.fonttype': 'none',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 10
})


def plot_histograms(data):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    sns.histplot(x='WM_tract', hue='Connection', data=data, multiple='dodge', shrink=0.8, palette='Paired')
    plt.subplot(1, 2, 2)
    sns.histplot(x='WM_tract', hue='Connection', data=data, multiple='fill', shrink=0.8, palette='Paired')
    plt.ylabel('Percentage')
    plt.tight_layout()
    return plt.gcf()

def plot_histograms_inv(data):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    sns.histplot(x='Connection', hue='WM_tract', data=data, multiple='dodge', shrink=0.8, palette='Paired')
    plt.subplot(1, 2, 2)
    sns.histplot(x='Connection', hue='WM_tract', data=data, multiple='fill', shrink=0.8, palette='Paired')
    plt.ylabel('Percentage')
    plt.tight_layout()
    return plt.gcf()


def ax_violon_box_strip(ax, x, y, data):
    palette = sns.light_palette("seagreen", as_cmap=True)
    sns.violinplot(x=x, y=y, data=data, dodge=False,
                   scale="width", inner=None, ax=ax, palette='Blues')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))

    sns.boxplot(x=x, y=y, data=data, saturation=1, showfliers=False,
                width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
    old_len_collections = len(ax.collections)
    sns.stripplot(x=x, y=y, data=data, dodge=False, ax=ax, color='k', alpha=0.1, s=1)
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def plot_violin_plots(data, metrics, labels):
    fig, axes = plt.subplots(1, len(metrics), figsize=(8, 3))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = ax_violon_box_strip(axes[i], 'WM_tract', metric, data)
        ax.set_ylabel(label)
    plt.tight_layout()
    return fig, axes


def plot_relationships(data, metrics, labels):
    fig, axes = plt.subplots(2, len(metrics), figsize=(8, 6))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        axes[0, i].hexbin(data['tract_dist'], data[metric], cmap='viridis', gridsize=20, bins='log')
        axes[0, i].set_xlabel('WM tract length')
        axes[0, i].set_ylabel(label)

        axes[1, i].hexbin(data['tract_num'], data[metric], cmap='viridis', gridsize=20, bins='log')
        axes[1, i].set_xlabel('WM tract count')
        axes[1, i].set_ylabel(label)
    return fig, axes


# Setup
subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL019", "EL020", "EL021", "EL022", "EL024",
         "EL026", "EL027", "EL028"]
sub_path = 'X:\\4 e-Lab\\'
path_save = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', 'Across', 'WM_tracts',
                         'Figures')
metrics = ['LL', 'delay', 'Sig']
labels = ['LL', 'CCEP onset [ms]', 'CCEP Probability']
con_all = ls.get_connections(subjs, sub_path)
con_all.loc[con_all.True_peak < 0.5, 'Sig'] = 0
con_all = con_all[con_all.Sig > -1].reset_index(drop=True)
# Main loop
n_fiber = 10
radia = [15]
radia = [15]
for start_r in radia:
    for end_r in radia:
        print(f"Processing for start_r={start_r} and end_r={end_r}")
        # Load and process data for each subject
        data_all = pd.DataFrame()
        for subj in subjs:
            print(subj, end='\r')
            file = os.path.join(sub_path, 'Patients', subj, 'Electrodes', 'Tracts',
                                f"{subj}_tracts_contacts_s{start_r}_e{end_r}_Region.csv")
            # self.
            if os.path.isfile(file):
                tract_matrix = pd.read_csv(file)
                data_subj = ls.adding_distance_tracts(con_all[con_all.Subj == subj].reset_index(drop=True),
                                                      tract_matrix, True)
                data_all = pd.concat([data_all, data_subj], ignore_index=True)
        # 1. adding Destreiux labels
        data_all = ls.adding_anatomy(data_all[
                                         ['Subj', 'Stim', 'Chan', 'H', 'Sig', 'DI', 'LL', 'd', 'delay', 'StimA',
                                          'ChanA',
                                          'tract_dist', 'tract_num']], pair=1, area='Destrieux')
        # 2. group connections into loca, direct, and indirect
        data_all = ls.group_connections(data_all)

        # 3. Categorize data into s+/s- and e+/e- for stuctural and effective connections
        data_all['WM_tract'] = data_all['tract_num'].apply(lambda x: 's+' if x > n_fiber else 's-')
        data_all['Connection'] = data_all['Sig'].apply(lambda x: 'e-' if x == 0 else 'e+')
        # 4. remove connections with unknown WM tracts
        tract_path = 'X:\\4 e-Lab\EvM\Projects\EL_experiment\Analysis\Patients\Across\WM_tracts'
        dist_l = pd.read_csv(os.path.join(tract_path, 'mni_destrieux_distances_l.csv'), index_col=0)
        non_nan_rows = dist_l.dropna(how='all', axis=0)  # Drops rows where all values are NaN
        d_atlas_labels = non_nan_rows.index.tolist()
        # 5. Filter data that are not local and are in destrieux labels known to e in at least one WM atlas
        data_plot = data_all[(data_all.Group != 'local') &(data_all.H == 0) &
                             np.isin(data_all.ChanR, d_atlas_labels) & np.isin(data_all.StimR,
                                                                               d_atlas_labels)].reset_index(drop=True)
        # Plot figures
        data_plot_plus = data_plot[(data_plot['Connection'] == 'e+') & (data_plot['WM_tract'] == 's+')].reset_index(
            drop=True)
        fig1 = plot_histograms(data_plot)
        plt.suptitle(
            f"Connection distribution with sending radius = {start_r}mm and receiving radius ={end_r}mm (min {n_fiber} fibers)")
        plt.tight_layout()
        # fig1.savefig(os.path.join(path_save, f"Fig1_s{start_r}_e{end_r}.svg"))
        fig1.savefig(os.path.join(path_save, f"Fig1_s{start_r}_e{end_r}_{n_fiber}f.svg"))

        fig2, _ = plot_violin_plots(data_plot[(data_plot['Connection'] == 'e+')], metrics, labels)
        plt.suptitle(f"CCEP metrics with sending radius = {start_r}mm and receiving radius ={end_r}mm (min 50 fibers)")
        plt.tight_layout()
        fig2.savefig(os.path.join(path_save, f"Fig2_s{start_r}_e{end_r}_{n_fiber}f.png"), dpi=300)

        fig3, _ = plot_relationships(data_plot_plus, metrics, labels)
        plt.suptitle(
            f"CCEP metrics by WM tracts with sending radius = {start_r}mm and receiving radius ={end_r}mm (min 50 fibers)")
        plt.tight_layout()
        fig3.savefig(os.path.join(path_save, f"Fig3_s{start_r}_e{end_r}_{n_fiber}f.png"), dpi=300)

        print(f"Saved figures for start_r={start_r} and end_r={end_r}")
