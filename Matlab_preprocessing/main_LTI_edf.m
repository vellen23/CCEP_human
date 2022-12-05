clear all
close all
%% Windows
% not a nice way to change whether I'm working on my mac or insel-pc
cwp         = pwd;
sep         = '\';
idcs        = strfind(cwp,sep);
path        = cwp(1:idcs(end)-1);
% addpath([path '\toolboxes\fieldtrip']); % not sure if you need it though ... it's not bad to have it
idcs        = strfind(path,sep);
path        = path(1:idcs(end)-1);  % path 0, where all important folders are (Patients, codes, etc.)
%addpath('C:\Program Files\MATLAB\R2020b\toolbox\fieldtrip');
addpath(genpath([path '\elab\Epitome'])); % maxime's software to open EEG data also available on github

clearvars cwp idcs
addpath([pwd '/nx_preproc']);
% ft_defaults;
warning('off','MATLAB:xlswrite:AddSheet'); %optional

%% patient specific

subj            = 'EL018'; %% change name if another data is used !!
% path_patient    = ['Y:\eLab\Patients\' subj];   
path = 'Y:\eLab\Patients\';
path_patient    = [path,  subj];  
dir_files       = [path_patient,'/data_raw/LT_Experiment'];
% load([path_patient,'\infos\BP_label.mat']); % table with the bipolar labels and hwo they are called in MP edf files
% dir_files       = [path_patient,'\data_raw\LT_experiment'];% folder where raw edf are stored
% load labels
MP_label = importfile_MPlabels([path_patient '\infos\' subj '_lookup.xlsx'], 'Channels');
BP_label = importfile_BPlabels([path_patient '\infos\' subj '_lookup.xlsx'], 'Channels_BP');

BP_label = BP_label(~isnan(BP_label.chan_BP_P),:);
MP_label= MP_label(~isnan(MP_label.Natus),:);


%% 1. log 

log_files= dir([dir_files '\*.log']);
i = 2; % find automated way or select manually
log             = import_logfile([dir_files '\' log_files(i).name]);
stimlist_all   = read_log(log);
% based on the log file, the function creates a table with each stimulation
% (row) and the parameters (columns (time, intensity, channels, protocol)
protocols = ["LTD1","LTD10","LTP50"];
% stimlist = stimlist_all(stimlist_all.type
%% 2. file
% select only the stimulation of desired protocol
data_files= dir([dir_files '\*.EDF']);
i = 1;
filepath               = [dir_files '\' data_files(i).name]; % the file you want to open 
H                      = Epitome_edfExtractHeader(filepath);
[hdr_edf, EEG_all]     = edfread_data(filepath);
%% protcol specific stimulation table
prot = "LTP50"; %data_files(i).name(7:end-8); % selected protocol of this data: ["LTD1","LTD10","LTP50"];
% find stimulations of cselected protocol
stimlist = stimlist_all(startsWith(string(stimlist_all.type), prot),:); % change to LTP1, LTD10, LTD50 for you protocol
%% 3. trigger
% [hdr_edf, trig]     = edfread(filepath,'targetSignals','TRIG'); %TeOcc5, TRIG
c_trig         = find(hdr_edf.label==string('TRIG'));
trig           = EEG_all(c_trig,:); %  trigger data same size as a EEG channel. consits of zeros except for the stimulation sample
Fs             = round(hdr_edf.frequency(1)); % recording Fs
% [pks,locs]   = findpeaks(trig_CR1,'MinPeakDistance',2*Fs,'Threshold',0.9,'MaxPeakWidth', 0.002*Fs);
[pks,TTL_sample]     = findpeaks(trig,'MinPeakDistance',1*Fs);
TTL_sample           = TTL_sample';
%% add column TTL to your stimlist
stimlist.TTL = zeros(height(stimlist),1);
% find an easy way to align one TTL_sample to a stimulation you are sure
% they match. usually first stimulation with first trigger
i =1;
stimlist.TTL(i)= TTL_sample(1);
%% quick fix, find peaks of specific channel to find TTL without triggers
% needs to be improved
[pk, locs_stim] = findpeaks(nanmean(EEG_all(75:80,:),1), Fs, 'MinPeakDistance',0.5);
locs_stim = round(locs_stim*Fs)-7;

%% 4. find the TTL_sample for the remaining stimulations based on expected sample and timing
% timestamp of selected stimulation with known TTL
ts1              = stimlist.h(i)*3.6e3+stimlist.min(i)*60+stimlist.s(i)+stimlist.us(i)/1000000; 
size_log        = size(stimlist);
ttl_1            = stimlist.TTL(i);% TTL1(1);
day  =0;
TTL_copy = TTL_sample;
locs_copy = locs_stim;
for s = 1: size(stimlist,1)
    if stimlist.date(s)<stimlist.date(i)
        day = -24;
    elseif stimlist.date(s)>stimlist.date(i)
        day = 24;
    else
        day = 0;
    end
    % calculate expected timestamp based on timing
    timestamp              = ((stimlist.h(s)+day)*3.6e3+stimlist.min(s)*60+stimlist.s(s)+stimlist.us(s)/1000000);
    sample_cal             = (timestamp-ts1)*Fs+ttl_1; %expected TTL 
    [ d, ix ]              = min(abs(TTL_copy-sample_cal)); % closes real TTL to expected TTL
    %[ d, ix ] = min( abs( round(timestamp-ts1)+ts0-double(TTL_table.timestamp)) );
    if d < 0.5*Fs 
        stimlist.TTL(s)   = TTL_copy(ix); 
        TTL_copy(ix) = - max(TTL_copy); % remove used TTL that they don't appear several times
        stimlist.noise(s)   = 0;
    else % if no TTL was found, just keep the calculated one. it is marked as "noise" 
        [ d, ix ]              = min(abs(locs_copy-sample_cal));
        if d< 0.5*Fs
            stimlist.TTL(s)   = locs_copy(ix); 
            locs_copy(ix) = - max(locs_copy); % remove used TTL that they don't appear several times
            stimlist.noise(s)   = 1;
        else
            stimlist.TTL(s)     = round(sample_cal);
            stimlist.noise(s)   = 2;
        end
    end

end
disp('TTL aligned');
%%
prot = "LTD1";
stimlist = stimlist_LT3(startsWith(string(stimlist_LT3.type), prot),:); % change to LTP1, LTD10, LTD50 for you protocol
%% 5. Test trigger, plot some stimulations
clf(figure(1))
Fs     = hdr_edf.frequency(1);
n_trig = 10;
c= 117;
t      = stimlist.TTL(n_trig);
IPI    = stimlist.IPI_ms(n_trig);
x_s = 10;
x_ax        = -x_s:1/Fs:x_s;

plot(x_ax,EEG_all(c,t-x_s*Fs:t+x_s*Fs));
%hold on
%plot(x_ax,EEG_mean(1,t-x_s*Fs:t+x_s*Fs));
xline(0, '--r');
xline(IPI/1000, '--r');


%% 6. get bipolar montage of EEG
% load BP_labels and find the index on how the channels are stored in the edf files 
ix              = find_BP_index(hdr_edf.label', BP_label.labelP_EDF, BP_label.labelN_EDF);
pos_ChanP       =  ix(:,1);
pos_ChanN       =  ix(:,2);
EEG_BP          = EEG_all(pos_ChanN,:)-EEG_all(pos_ChanP,:);
%%
clf(figure(1))
Fs     = hdr_edf.frequency(1);
n_trig = 34;
c= 92;
t      = stimlist.TTL(n_trig);
IPI    = stimlist.IPI_ms(n_trig);
x_s = 10;
x_ax        = -x_s:1/Fs:x_s;

plot(x_ax,EEG_BP(c,t-x_s*Fs:t+x_s*Fs));
%hold on
%plot(x_ax,EEG_mean(1,t-x_s*Fs:t+x_s*Fs));
xline(0, '--r');
xline(IPI/1000, '--r');

%% Cut in block
path_preprocessed = ['Y:\eLab\Patients\' subj '\Data\LT_experiment'];
prot_num = 2;
cut_block_edf_LT(EEG_BP, stimlist,prot,prot_num, Fs, subj, BP_label, path_preprocessed)
%(EEG_block, stim_list,type,block_num, Fs, path_patient, subj,BP_label, path_pp)

