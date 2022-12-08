%% RERUN preprocess
clear all
close all
%% UPDATE STIMLIST and TTL 
subj            = 'EL018';
%block_path = ['Y:\eLab\Patients\' subj '\Data\LT_experiment\data_blocks'];
block_path = ['Y:\eLab\Patients\' subj '\Data\EL_experiment\experiment1\data_blocks'];

%% patient specific
path = 'Y:\eLab\Patients\';
subj            = 'EL018'; %% change name if another data is used !!
path_patient    = [path,  subj];  
dir_files       = [path_patient,'/data_raw/EL_Experiment'];
% load([path_patient,'\infos\BP_label.mat']); % table with the bipolar labels and hwo they are called in MP edf files
% dir_files       = [path_patient,'\data_raw\LT_experiment'];% folder where raw edf are stored
% load labels
MP_label = importfile_MPlabels([path_patient '\infos\' subj '_lookup.xlsx'], 'Channels');
BP_label = importfile_BPlabels([path_patient '\infos\' subj '_lookup.xlsx'], 'Channels_BP');
BP_label= BP_label(~isnan(BP_label.chan_BP_N),:);
MP_label= MP_label(~isnan(MP_label.Natus),:);

%% get rescale factor based on one block during 5min baseline
% load one file, EEG and scalpEEG
if exist( ['Y:\eLab\Patients\' subj '\Data\EL_experiment\experiment1\data_blocks\scale_fac.mat'], 'file')
    load(['Y:\eLab\Patients\' subj '\Data\EL_experiment\experiment1\data_blocks\scale_fac.mat'])
else
    [sclA, sclC]             = get_rescale_factors(EEG, Fs, 0, 250);
    [sclA_scalp, sclC_scalp] = get_rescale_factors(scalpEEG, scalpFs, 1, 250);
    save(['Y:\eLab\Patients\' subj '\Data\EL_experiment\experiment1\data_blocks\scale_fac.mat'],'sclA','sclA_scalp', 'sclC','sclC_scalp');
end
% save in Data folder ! 
%% 

% % path where all blocks are stored
% % block_path     = uigetdir(['T:\EL_experiment\Patients\', subj, '\Data\EL_experiment']); %
block_files     = dir(block_path);
isdir           = [block_files.isdir]; % Get all the codes
block_files     = block_files(isdir==1); % Select only the p and H codes, delete the rest
for i=3:length(block_files)
    disp(block_files(i).name);
    run_pp_check =1;
    if exist(char([block_path, '\', block_files(i).name, '\', 'ppEEG.mat']), 'file')==2
        pp_date = dir(char([block_path, '\', block_files(i).name, '\', 'ppEEG.mat'])).datenum;
        raw_date = dir(char([block_path, '\', block_files(i).name, '\', block_files(i).name,'.mat'])).datenum;
        if pp_date>raw_date
            run_pp_check = 0;
        end
    end
    if run_pp_check
        run_pp(char([block_path, '\', block_files(i).name]), sclA, sclC );
        run_pp_scalp(char([block_path, '\', block_files(i).name]), sclA_scalp, sclC_scalp, BP_label);
    % %     
        %sanity_checks(char([block_path, sep, block_files(i).name]),BP_label );
    %     %save_metadata(char([block_path, sep, block_files(i).name]));
    %     %rescale(char([block_path, sep, block_files(i).name]),sclA, sclC)
        create_TTL(char([block_path, '\', block_files(i).name]));
        score2list(char([block_path, '\', block_files(i).name]),0);
    end
end