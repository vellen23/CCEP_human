import os
import numpy as np
import pandas as pd
import pathpy as pp
import basic_func as bf
from scipy import stats

def con_sleep_stats(con_trial):
    con_trial['Con_ID'] = con_trial.groupby(['Stim', 'Chan']).ngroup()
    if "SleepState" not in con_trial:
        con_trial = bf.add_sleepstate(con_trial)
    con_trial['LL_sig'] = con_trial['LL'] * con_trial['Sig']
    # clean trials
    con_trial = con_trial[(con_trial.Sig > -1) & (con_trial.Artefact < 1)].reset_index(drop=True)

    # Define the baseline condition
    baseline_condition = 'Wake'
    other_conditions = ['NREM',
                        'REM']  # [cond for cond in con_trial['SleepState'].unique() if cond != baseline_condition]

    # Pre-group the data by Con_ID and SleepState for faster lookup
    grouped_data = con_trial.groupby(['Con_ID', 'Stim', 'Chan', 'SleepState'])['LL_sig']

    results = []
    for con_id in con_trial['Con_ID'].unique():
        stim = con_trial.loc[con_trial.Con_ID == con_id, 'Stim'].values[0].astype('int')
        chan = con_trial.loc[con_trial.Con_ID == con_id, 'Chan'].values[0].astype('int')

        # If baseline data for the current connection ID is not present, skip this iteration
        if (con_id, stim, chan, baseline_condition) not in grouped_data.groups:
            continue

        baseline_data = grouped_data.get_group((con_id, stim, chan, baseline_condition)).values
        if np.nanmean(baseline_data) == 0:
            continue
        # Iterate over other conditions (excluding the baseline)
        for condition in other_conditions:
            # If data for the current condition and connection ID is not present, skip this iteration
            if (con_id, stim, chan, condition) not in grouped_data.groups:
                continue

            current_data = grouped_data.get_group((con_id,stim, chan, condition)).values
            n1, n2 = len(baseline_data), len(current_data)
            # Perform a t-test
            # t_stat, p_value = stats.ttest_ind(baseline_data, current_data, equal_var=False)
            u_stat, p_value = stats.mannwhitneyu(baseline_data, current_data)
            rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
            # Calculate Effect size
            # pooled_std = np.sqrt(
            #     ((n1 - 1) * np.var(baseline_data, ddof=1) + (n2 - 1) * np.var(current_data, ddof=1)) / (
            #                 n1 + n2 - 2))
            # cohen_d = abs((np.mean(baseline_data) - np.mean(current_data)) / pooled_std)

            # Check if the p-value is less than 0.05 to determine significance
            significant = p_value < 0.05

            # Append results to the placeholder list
            results.append((stim, chan, condition, rank_biserial, significant))

    # Convert the results list to a DataFrame
    con_sleep = pd.DataFrame(results, columns=['Stim', 'Chan', 'SleepState', 'biserial', 'Sig'])
    return con_sleep
def node_features_sleep(con_trial, file):
    sleepstates = ['Wake', 'NREM', 'REM']
    if "SleepState" not in con_trial:
        con_trial = bf.add_sleepstate(con_trial)
    con_trial['LL_sig'] = con_trial['LL'] * con_trial['Sig']
    # clean trials
    con_trial = con_trial[(con_trial.Sig > -1) & (con_trial.Artefact < 1)].reset_index(drop=True)
    con_trial_sleep = con_trial.groupby(['Stim', 'Chan', 'SleepState'], as_index=False)[['Sig', 'LL_sig']].mean()
    con_trial_sleep = con_trial_sleep[np.isin(con_trial_sleep.SleepState, sleepstates)].reset_index(
        drop=True)
    #  os.makedirs(exp_dir, exist_ok=True)
    # file = os.path.join(exp_dir, 'node_features_sleep.csv')
    # Calculate metrics for each node
    nodes = []
    Dout = []
    Din = []
    Dout_w = []
    Din_w = []
    Cb = []
    Cc = []
    SS = []
    # repeat for each condition
    for ss in sleepstates:
        df_sleep = con_trial_sleep[(con_trial_sleep.Sig > 0) & (con_trial_sleep.SleepState == ss)].reset_index(
            drop=True)
        # Create a directed network from the DataFrame
        n = pp.Network(directed=True)
        for index, row in df_sleep.iterrows():
            n.add_edge(str(row['Stim']), str(row['Chan']), weight=row['LL_sig'])

        for node in n.nodes:
            nodes.append(node)
            Dout.append(n.nodes[node]['outdegree'])
            Din.append(n.nodes[node]['indegree'])
            Dout_w.append(n.nodes[node]['outweight'])
            Din_w.append(n.nodes[node]['inweight'])
            Cb.append(pp.algorithms.centralities.betweenness(n)[node])
            Cc.append(pp.algorithms.centralities.closeness(n)[node])
            SS.append(ss)

    # Create DataFrame
    df = pd.DataFrame({
        'Chan': nodes,
        'Dout': Dout,
        'Din': Din,
        'Dout_w': Dout_w,
        'Din_w': Din_w,
        'Cb': Cb,
        'Cc': Cc,
        'SleepState': SS
    })
    df.Chan = df.Chan.astype('int')
    # df.to_csv(file, header=True, index=False)
    return df