%% RERUN preprocess
clear all
close all

%% get rescale factor based on one block during 5min baseline
% load one file, EEG and scalpEEG
if exist( [path_patient '\Data\EL_experiment\experiment1\data_blocks\scale_fac.mat'], 'file')
    load([path_patient '\Data\EL_experiment\experiment1\data_blocks\scale_fac.mat'])
else
    [sclA, sclC]             = get_rescale_factors(EEG, Fs, 0, 30);
    % [sclA_scalp, sclC_scalp] = get_rescale_factors(scalpEEG, scalpFs, 1, 30);
    save([path_patient '\Data\EL_experiment\experiment1\data_blocks\scale_fac.mat'],'sclA','sclA_scalp', 'sclC','sclC_scalp');
end

%% 

% path where all blocks are stored
block_path     = uigetdir([path_patient, '\Data\Pharmaco']); %
block_files     = dir(block_path);
isdir           = [block_files.isdir]; % Get all the codes
block_files     = block_files(isdir==1); % Select only the p and H codes, delete the rest

%% 
for i=3:length(block_files)
    disp(block_files(i).name);

    run_pp(char([block_path, '\', block_files(i).name]), sclA, sclC );
    if exist(char([block_path, '\', block_files(i).name, '\', 'scalpEEG.mat']), 'file')==2
        run_pp_scalp(char([block_path, '\', block_files(i).name]), sclA_scalp, sclC_scalp, BP_label);
    end

    create_TTL(char([block_path, '\', block_files(i).name]));

end