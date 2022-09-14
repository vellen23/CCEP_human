close all; clear all;

%% Windows, load toolboxes
cwp         = pwd;
sep         = '\';
idcs        = strfind(cwp,sep);
path        = 'T:\EL_experiment\Codes';
addpath([path '\toolboxes\fieldtrip']);
idcs        = strfind(path,sep);
path        = path(1:idcs(end)-1);  % path 0, where all important folders are (Patients, codes, etc.)
%addpath('C:\Program Files\MATLAB\R2020b\toolbox\fieldtrip');
addpath(genpath([path '\elab\Epitome']));
addpath(genpath([path '\toolboxes\nx_toolbox']));

clearvars cwp idcs
addpath([pwd '/nx_preproc']);
ft_defaults;
warning('off','MATLAB:xlswrite:AddSheet'); %optional

%% patient specific
subj            = 'EL015'; %% change name if another data is used !!
path_patient    = [path, '/Patients/' subj];  
dir_files       = [path_patient,'/data_raw/EL_Experiment'];
load([path_patient,'\infos\BP_label.mat']); % table with the bipolar labels and hwo they are called in MP edf files

%% 1. log 
% file_log =[dir_files '/20220622_EL015_log.log'];
file_log = 'T:\EL_experiment\Patients\EL015\infos\20220622_EL015_log.log';
log                 = import_logfile(file_log);
stimlist_all   = read_log(log);
stimlist_all = stimlist_all(stimlist_all.type~='Clinic',:);
stimlist_all.keep = ones(height(stimlist_all),1);
% for LTP/LTD select specific type
stop
%% type
type                = 'CR';

  %%
files = dir(dir_files);
for j=13:length(files)
    %% 1. read first raw data
    file = files(j).name;
    filepath               = [dir_files '/' file]; %'/Volumes/EvM_T7/EL008/Data_raw/EL008_BM_1.edf';
    H                      = Epitome_edfExtractHeader(filepath);
    % [hdr_edf, EEG_all]     = edfread_data(filepath);
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
    ix_startblock  = find(diff(locs)/Fs>200); 
    % find trigger that is starting a new block (first trigger that had a
    % distance of 5min after the last one
    TTL_startblock = locs(ix_startblock(1)+1);
    %% 3. assign selected trigger to stimlist
    stimlist.TTL = zeros(height(stimlist),1);
    blocks = unique(stimlist.stim_block);
    ix_block = find(stimlist.stim_block==blocks(3));
    stimlist(ix_block(1),'TTL')= {TTL_startblock};
    %% 4. for each stimulation, assign the expected TTL 
    % if len stimtable and and # f triggers are identical, we can just merge
    i                = ix_block(1); % selected one where you are sure hte trigger is correct
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
        if d < 1*Fs
            stimlist.TTL(s)   = locs_copy(ix); 
            locs_copy(ix) = - max(locs_copy);
            stimlist.noise(s)   = 0;
        else
            stimlist.TTL(s)     = round(sample_cal);
            stimlist.noise(s)   = 1;
        end

    end
    stim_cut = find(diff(find(stimlist.noise==1))>1);
    stim_noise = find(stimlist.noise==1);
    if stimlist.noise(end) == 0 
        stimlist.noise(height(stimlist)+1) = 1;
        stimlist_all.noise(height(stimlist_all)+1) = 1;
        stim_cut = find(diff(find(stimlist.noise==1))>1);
        stim_noise = find(stimlist.noise==1);   
    end
    if stimlist.noise(1) == 0 
        stim_noise(1) = 0;
    end
    stimlist_all(1:stim_noise(stim_cut(end)+1)-1,"keep") = {0};
    stimlist_all = stimlist_all(stimlist_all.keep==1,:);
    stimlist = stimlist(stim_noise(stim_cut(1))+1:stim_noise(stim_cut(end)+1)-1,:);
    disp('TTL aligned');

    %% 5. Test trigger
    clf(figure(1))
    Fs     = hdr_edf.frequency(1);
    %Fs = 148;
    n_trig = 174;
    t      = stimlist.TTL(n_trig);
    IPI    = stimlist.IPI_ms(n_trig);
    x_s = 10;
    x_ax        = -x_s:1/Fs:x_s;
    c= 80;
    plot(x_ax,EEG_all(c,t-x_s*Fs:t+x_s*Fs));
    hold on
    plot(x_ax,trig(1,t-x_s*Fs:t+x_s*Fs));
    xline(0, '--r');
    xline(IPI/1000, '--r');


    %% 6. get bipolar montage of EEG
    % bipolar
    ix        = find_BP_index(hdr_edf.label', BP_label.labelP, BP_label.labelN);
    pos_ChanP =  ix(:,1);
    pos_ChanN =  ix(:,2);
    % EEG_all         = [EEG_all; zeros(1,size(EEG_all,2))];
    EEG_block       = EEG_all(pos_ChanN,:)-EEG_all(pos_ChanP,:);
    
    %% 7.1 CR loop for cutting blocks
    
    stimlist.b = stimlist.stim_block;
    blocks          = unique(stimlist.b);
    if height(stimlist(stimlist.b==blocks(3),:))>height(stimlist(stimlist.b==blocks(2),:))+20
        ix = find(stimlist.stim_block==blocks(3),1);
        stimlist.b(1:ix-2) = blocks(3);
        blocks          = unique(stimlist.b); %
    end

    for i=2:length(blocks)%2:11
        block_num               = blocks(i);
        stim_list           = stimlist(stimlist.b==block_num,:);%   
        cut_block_edf(EEG_block, stim_list,type,block_num, Fs, path_patient, subj, BP_label)
    end
    assignin('base',['stimlist' file(end-7:end-4)], stimlist)
end

%%%%%
%% update cutting with having the stimlist already prepared
for j=2:length(files)
    file = files(j).name;
    disp(file);
    filepath               = [dir_files '/' file];
    VariableName=['stimlist' file(end-7:end-4)];
    eval(['stimlist = ',VariableName,';']);
    H                      = Epitome_edfExtractHeader(filepath);
    [hdr_edf, EEG_all]     = edfread_data(filepath);
    Fs     = hdr_edf.frequency(1);
    %% 6. get bipolar montage of EEG
    % bipolar
    ix        = find_BP_index(hdr_edf.label', BP_label.labelP, BP_label.labelN);
    pos_ChanP =  ix(:,1);
    pos_ChanN =  ix(:,2);
    % EEG_all         = [EEG_all; zeros(1,size(EEG_all,2))];
    EEG_block       = EEG_all(pos_ChanN,:)-EEG_all(pos_ChanP,:);
    clear EEG_all
    %% 7.1 CR loop for cutting blocks
    blocks          = unique(stimlist.b);
    for i=2:length(blocks)%2:11
        block_num               = blocks(i);
        stim_list           = stimlist(stimlist.b==block_num,:);%   
        cut_block_edf(EEG_block, stim_list,type,block_num, Fs, path_patient, subj, BP_label)
    end
end
