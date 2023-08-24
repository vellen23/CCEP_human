close all; clear all;

% %% Windows, load toolboxes
% cwp         = pwd;
% sep         = '\';
% idcs        = strfind(cwp,sep);
% path        = 'T:\EL_experiment\Codes';
% %addpath([path '\toolboxes\fieldtrip']);
% idcs        = strfind(path,sep);
% path        = path(1:idcs(end)-1);  % path 0, where all important folders are (Patients, codes, etc.)
% %addpath('C:\Program Files\MATLAB\R2020b\toolbox\fieldtrip');
% % addpath(genpath([path '\elab\Epitome']));
% addpath(genpath([path '\toolboxes\nx_toolbox']));
% 
% clearvars cwp idcs
addpath([pwd '/nx_preproc']);
% % ft_defaults;
% warning('off','MATLAB:xlswrite:AddSheet'); %optional

%% patient specific
path = 'Y:\eLab\Patients\';
path = 'X:\\4 e-Lab\\Patients\\';
subj            = 'EL027'; %% change name if another data is used !!
path_patient    = [path,  subj];  
dir_files       = [path_patient,'/data_raw/EL_Experiment'];

%% 1. log 
log_files= dir([dir_files '\*BM*.log']);
i = 1; % find automated way or select manually
log             = importfile_log_2([dir_files '\' log_files(i).name]);
stimlist_all = log(log.date~="WAIT",:);
% stimlist_all = stimlist_all(stimlist_all.type=="BMCT",:);
% stimlist_all = stimlist_all(stimlist_all.type=="BMCT",:);
% file_log =[dir_files '/20220622_EL015_log.log'];
% file_log = 'T:\EL_experiment\Patients\EL016\infos\20220921_EL016_log.log';
% log                 = import_logfile(file_log);
% stimlist_all   = read_log(log);
stimlist_all.Properties.VariableNames{8} = 'stim_block';
stimlist_all.Properties.VariableNames{2} = 'h';
stimlist_all.keep = ones(height(stimlist_all),1);
stimlist_all.date = double(stimlist_all.date);
date = 20230822;
midnight = find(stimlist_all.h==0);
if ~isempty(midnight)
    midnight = midnight(1);
    stimlist_all.date(1:midnight) = date;
    stimlist_all.date(midnight:end) = date+1;
else
    stimlist_all.date(:) = date;
end
%% update block number
n_block = 1;
stimlist_all.stim_block = stimlist_all.stim_block+n_block;

%% type
type                = 'BM';
path_pp = [path_patient '\Data\EL_experiment\experiment1'];

%% 
dir_files       = [path_patient,'/data_raw/EL_Experiment'];
files= dir([dir_files '\*BM*.EDF']);% dir([dir_files '\*BM*.EDF']);
for j=1:length(files)
    %% 1. read first raw data
    file = files(j).name
    filepath               = [dir_files '/' file]; %'/Volumes/EvM_T7/EL008/Data_raw/EL008_BM_1.edf';
    H                      = Epitome_edfExtractHeader(filepath);
    [hdr_edf, EEG_all]     = edfread_data(filepath);
    stimlist = stimlist_all;
    stimlist = removevars(stimlist, 'keep');
    %% 2. find triggers
    % [hdr_edf, trig]     = edfread(filepath,'targetSignals','TRIG'); %TeOcc5, TRIG
    c_trig         = find(hdr_edf.label=="TRIG");
    trig           = EEG_all(c_trig,:);
    Fs             = round(hdr_edf.frequency(1));
    % [pks,locs]   = findpeaks(trig_CR1,'MinPeakDistance',2*Fs,'Threshold',0.9,'MaxPeakWidth', 0.002*Fs);
    [pks,locs]     = findpeaks(trig,'MinPeakDistance',1*Fs);
    locs           = locs';
    ix_startblock  = find(diff(locs)/Fs>100); 
    % find trigger that is starting a new block (first trigger that had a
    % distance of 5min after the last one
    stimlist.TTL = zeros(height(stimlist),1);
    if isempty(ix_startblock)
        TTL_startblock = locs(1);
        i = input('enter index of first trigger: '); % seelect manually (first trigger to stimlist)
    else  
        TTL_startblock = locs(ix_startblock(1)+1);
        blocks = unique(stimlist.stim_block);
        ix_block = find(stimlist.stim_block==blocks(2));
        i                = ix_block(1);
        
    end
    stimlist(i,'TTL')= {TTL_startblock};
    %% 4. for each stimulation, assign the expected TTL 
    % if len stimtable and and # f triggers are identical, we can just merge
    % i                = ix_block(1); % selected one where you are sure hte trigger is correct
    ts1              = stimlist.h(i)*3.6e3+stimlist.min(i)*60+stimlist.s(i)+stimlist.us(i)/1000000; 
    size_log        = size(stimlist);
    % enter manually
    ttl0            = stimlist.TTL(i);% TTL1(1);
    day  =0;
    locs_copy = locs;
    for s = 1: size(stimlist,1)
        if stimlist.date(s)<stimlist.date(i)
            day = -24;
        elseif stimlist.date(s)>stimlist.date(i)
            day = 24;
        else
            day = 0;
        end

        timestamp              = ((stimlist.h(s)+day)*3.6e3+stimlist.min(s)*60+stimlist.s(s)+stimlist.us(s)/1000000);
        sample_cal             = (timestamp-ts1)*Fs+ttl0; %expected TTL 
        [ d, ix ]              = min(abs(locs_copy-sample_cal));
        %[ d, ix ] = min( abs( round(timestamp-ts1)+ts0-double(TTL_table.timestamp)) );
        if d < 2*Fs
            stimlist.TTL(s)   = locs_copy(ix); 
            locs_copy(ix) = - max(locs_copy);
            stimlist.noise(s)   = 0;
        else
            stimlist.TTL(s)     = round(sample_cal);
            stimlist.noise(s)   = 1;
        end

    end
    
    %% 5. Test trigger
    clf(figure(1))
    Fs     = hdr_edf.frequency(1);
    %Fs = 148;
    n_trig = 20;
    t      = stimlist.TTL(n_trig);
    IPI    = stimlist.IPI_ms(n_trig);
    x_s = 3;
    x_ax        = -x_s:1/Fs:x_s;
    c = 70;%stimlist.ChanP(n_trig);
    plot(x_ax,EEG_all(c,t-x_s*Fs:t+x_s*Fs));
    hold on
    plot(x_ax,trig(1,t-x_s*Fs:t+x_s*Fs));
    xline(0, '--r');
    xline(IPI/1000, '--r');


    %% 6. get bipolar montage of EEG
    % bipolar
    ix        = find_BP_index(hdr_edf.label', BP_label.labelP_EDF, BP_label.labelN_EDF);
    pos_ChanP =  ix(:,1);
    pos_ChanN =  ix(:,2);
    % EEG_all         = [EEG_all; zeros(1,size(EEG_all,2))];
    EEG_all       = EEG_all(pos_ChanN,:)-EEG_all(pos_ChanP,:);
    
    %% 7.1 CR loop for cutting blocks
    stimlist = stimlist(stimlist.type~="BM_triplet",:);
    stimlist.b = stimlist.stim_block;
    blocks          = unique(stimlist.b);
    if height(stimlist(stimlist.b==blocks(2),:))>height(stimlist(stimlist.b==blocks(1),:))+20
        ix = find(stimlist.stim_block==blocks(2),1);
        stimlist.b(1:ix-1) = blocks(2);
    end
    %%
    blocks          = unique(stimlist.b);
    blocks = blocks(blocks>0);
    for i=1:length(blocks)%2:11
        block_num               = blocks(i);
        stim_list           = stimlist(stimlist.b==block_num,:);%   
        cut_block_edf(EEG_all, stim_list,'BM',block_num, Fs, subj, BP_label,path_pp)
        % cut_block_edf(EEG_block, stim_list,type,block_num, Fs, subj,BP_label, path_pp)
        %(EEG_block, stim_list,type,block_num, Fs, subj,BP_label, path_pp)
    end
    assignin('base',['stimlist' file(end-8:end-4)], stimlist)
    if height(stimlist_all)<10
        log_ix = log_ix+1;
        log             = import_logfile([dir_files '\' log_files(log_ix).name]);
        % file_log =[dir_files '/20220622_EL015_log.log'];
        % file_log = 'T:\EL_experiment\Patients\EL016\infos\20220921_EL016_log.log';
        % log                 = import_logfile(file_log);
        stimlist_all   = read_log(log);
        stimlist_all=stimlist_all(startsWith(string(stimlist_all.type), "CR"),:);
        stimlist_all.keep = ones(height(stimlist_all),1);
        % update block numbers
      
        idx = find(stimlist_all.stim_block>0);
        stimlist_all.stim_block(idx) = stimlist_all.stim_block(idx)+max(stimlist.b);

    end
end

%%%%%
%% update cutting with having the stimlist already prepared
for j=2:length(files)
    file = files(j).name;
    disp(file);
    filepath               = [dir_files '/' file];
    %VariableName=['stimlist' file(end-7:end-4)];
    VariableName=['stimlist' file(end-8:end-4)];
    eval(['stimlist = ',VariableName,';']);
    H                      = Epitome_edfExtractHeader(filepath);
    [hdr_edf, EEG_all]     = edfread_data(filepath);
    Fs     = hdr_edf.frequency(1);
    %% 6. get bipolar montage of EEG
    % bipolar
%     ix        = find_BP_index(hdr_edf.label', BP_label.labelP_EDF, BP_label.labelN_EDF);
%     pos_ChanP =  ix(:,1);
%     pos_ChanN =  ix(:,2);
    %EEG_all         = [EEG_all; zeros(1,size(EEG_all,2))];
    EEG_all       = EEG_all(pos_ChanN,:)-EEG_all(pos_ChanP,:);

    %% 7.1 CR loop for cutting blocks
    blocks          = unique(stimlist.b);
    blocks = blocks(blocks>0);
    for i=7:length(blocks)%2:11
        block_num               = blocks(i);
        stim_list           = stimlist(stimlist.b==block_num,:);%   
        cut_block_edf(EEG_all, stim_list,type,block_num, Fs, subj, BP_label,path_pp)
        %(EEG_all, stim_list,'CR',block_num, Fs, subj, BP_label,path_pp)
    end
end
