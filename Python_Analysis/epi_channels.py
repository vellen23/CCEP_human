import pandas as pd
import os
import numpy as np

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def elab2destrieux(label):
    # area == 'Region' or 'Area'
    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\EvM\Projects\EL_experiment\Analysis\Patients\Across\elab_labels.xlsx'
    atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')
    if len(atlas.loc[atlas.Abbreviation == label, 'Destrieux']) > 0:
        label_dest = atlas.loc[atlas.Abbreviation == label, 'Destrieux'].values[0]
    else:
        print(label)
    return label_dest


def output_table_subj(input_df, label_sel='Area'):
    # Filter the dataframe to include only rows where 'include' is 1
    if 'include' in input_df:
        input_df = input_df[input_df['include'] == 1].reset_index(drop=True)
    # Initialize a dictionary to hold counts for SOZ, IED, Propagation, and Other
    label_counts = {'SOZ': {}, 'IED': {}, 'Propagation': {}, 'Other': 0}

    # Populate the dictionary with counts for each label and feature
    for index, row in input_df.iterrows():
        label = row[label_sel]
        label = elab2destrieux(label.replace(" ", ""))
        if ~ np.isin(label.upper(), ['WM', 'OUT', 'PUTAMEN', 'UNKNOWN']):
            if row['x'] < 0:
                label = 'L_' + label
            else:
                label = 'R_' + label
            if row['SOZ'] == 1:
                label_counts['SOZ'][label] = label_counts['SOZ'].get(label, 0) + 1
            if row['IED'] == 1:
                label_counts['IED'][label] = label_counts['IED'].get(label, 0) + 1
            if row['Propagation'] == 1:
                label_counts['Propagation'][label] = label_counts['Propagation'].get(label, 0) + 1
            if row['SOZ'] == 0 and row['IED'] == 0 and row['Propagation'] == 0:
                # Increment count for 'Other' labels
                label_counts['Other'] += 1

    # Convert the dictionary to a dataframe format suitable for Excel output
    output_columns = ['Tissue-Type', 'Destrieux', 'Count']
    output_data = []

    for feature, labels in label_counts.items():
        if feature != 'Other':
            for label, count in labels.items():
                output_data.append([feature, label, count])
        else:
            output_data.append(['Uninvolved', 'Other', labels])  # 'Other' does not have a label

    output_df = pd.DataFrame(output_data, columns=output_columns)
    return output_df


def run_across_subj(subjs):
    df_all = []
    for ix_subj, subj in enumerate(subjs):
        path_lbls = os.path.join(sub_path, 'Patients', subj, 'Electrodes')
        lbls = pd.read_excel(os.path.join(path_lbls, subj + "_labels.xlsx"), header=0, sheet_name='BP')
        if 'type' in lbls:
            lbls = lbls[lbls.type == 'SEEG']
            lbls = lbls.reset_index(drop=True)

        output_df = output_table_subj(lbls)
        output_df.insert(0, 'Subj', subj)
        if ix_subj == 0:
            df_all = output_df.reset_index(drop=True)
        else:
            df_all = pd.concat([df_all, output_df]).reset_index(drop=True)
    output_file_path = os.path.join(sub_path, 'Patients', 'epileptic_channels.xlsx')
    with pd.ExcelWriter(output_file_path) as writer:
        df_all.to_excel(writer, index=False, sheet_name='Across')


## RUN
subjs = ["EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL019", "EL020", "EL021",
         "EL022", "EL024", "EL026", "EL027", "EL028"]

run_across_subj(subjs)
