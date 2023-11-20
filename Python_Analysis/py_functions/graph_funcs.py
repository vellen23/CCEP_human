import os
import numpy as np
import pandas as pd
import pathpy as pp
import basic_func as bf
from scipy import stats


def con_cond_stats(baseline_data, current_data, permutation=1, test='MWU'):
    """
    Calculates effect size and p-value for two distributions (1 connection different condition).

    Parameters:
    - baseline_data (array): The trial data (trial x time).
    - current_data (float): The width of the AUC window in seconds.
    - permutation (binary): If including a permutation test.
    - test (string): Decide on 'MWU' (Mann-Whitney U) or 'Cohen' (Cohen's D).

    Returns:
    - tuple: effect_size, p_value, p_permutation
    """
    n1, n2 = len(baseline_data), len(current_data)
    if test == 'MWU':
        # Perform a Mann-whitney
        u_stat, p_value = stats.mannwhitneyu(baseline_data, current_data)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)  # Effect size
    elif test == 'Cohen':
        combined_variance = ((n1 - 1) * np.var(baseline_data, ddof=1) + (n2 - 1) * np.var(current_data, ddof=1)) / (
                n1 + n2 - 2)
        # Perform a t-test (needed for p-value in Cohen's D)
        t_stat, p_value = stats.ttest_ind(baseline_data, current_data, equal_var=False)
        # Calculate Cohen's D
        effect_size = -(np.mean(baseline_data) - np.mean(current_data)) / np.sqrt(combined_variance)
    else:
        print('No valid statistical test selected')
        return np.nan, np.nan, np.nan

    # Permutation test
    if permutation:
        n_permutations = 200
        observed_difference = np.abs(np.mean(baseline_data) - np.mean(current_data))
        count_extreme_values = 0
        combined_data = np.hstack((baseline_data, current_data))
        for _ in range(n_permutations):
            np.random.shuffle(combined_data)
            permuted_baseline = combined_data[:n1]
            permuted_current = combined_data[n1:]

            permuted_difference = np.abs(np.mean(permuted_baseline) - np.mean(permuted_current))

            if permuted_difference >= observed_difference:
                count_extreme_values += 1

        p_permutation = count_extreme_values / n_permutations
    else:
        p_permutation = np.nan

    return effect_size, p_value, p_permutation


def con_sleep_stats(con_trial, sig=1):
    con_trial['Con_ID'] = con_trial.groupby(['Stim', 'Chan']).ngroup()
    if "SleepState" not in con_trial:
        con_trial = bf.add_sleepstate(con_trial)
    if sig:
        con_trial['LL_sig'] = con_trial['LL'] * con_trial['Sig']
    else:
        con_trial['LL_sig'] = con_trial['LL']
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

            current_data = grouped_data.get_group((con_id, stim, chan, condition)).values
            effect_size, p_value, p_permutation = con_cond_stats(baseline_data, current_data, 1,
                                                                 'MWU')  # con_cond_stats(baseline_data, current_data, permutation=1, test='MWU')
            effect_size_c, p_value_c, _ = con_cond_stats(baseline_data, current_data, 0, 'Cohen')
            # Append results to the placeholder list
            results.append((stim, chan, condition, effect_size, p_value, effect_size_c, p_value_c, p_permutation))

    # Convert the results list to a DataFrame
    con_sleep = pd.DataFrame(results,
                             columns=['Stim', 'Chan', 'SleepState', 'effect_size', 'p_MW', 'effect_size_C', 'p_C',
                                      'p_perm'])
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


def node_features_sleep_trial(con_trial, file):
    import networkx as nx
    sleepstates = ['Wake', 'NREM', 'REM']
    if "SleepState" not in con_trial:
        con_trial = bf.add_sleepstate(con_trial)
    con_trial['LL_sig'] = con_trial['LL'] * con_trial['Sig']
    # clean trials
    con_trial = con_trial[(con_trial.Sig > -1) & (con_trial.Artefact < 1)].reset_index(drop=True)

    # Initialize data structures for storing degree information
    degree_info = []

    # Process for each sleep state
    for ss in sleepstates:
        # Filter data for the current sleep state
        df_sleep = con_trial[con_trial['SleepState'] == ss]

        # Create a directed graph for each individual connection
        for _, row in df_sleep.iterrows():
            G = nx.DiGraph()
            G.add_edge(row['Stim'], row['Chan'], weight=row['LL'])

            # Calculate in- and out-degrees for each node
            for node in G.nodes():
                in_degree, out_degree = G.in_degree(node, weight='weight'), G.out_degree(node, weight='weight')
                degree_info.append({'Node': node, 'InDegree': in_degree, 'OutDegree': out_degree, 'SleepState': ss})

    # Convert the list of degree info to a DataFrame
    df_degrees = pd.DataFrame(degree_info)

    return df
