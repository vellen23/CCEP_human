close all; clear all;

%% Windows
cwp         = pwd;
idcs        = strfind(cwp,'\');
path        = cwp(1:idcs(end)-1);
idcs        = strfind(path,'\');
path        = path(1:idcs(end)-1);  % path 0, where all important folders are (Patients, codes, etc.)
%addpath('C:\Program Files\MATLAB\R2020b\toolbox\fieldtrip');
addpath(genpath(['T:\EL_experiment\Codes\Imaging\Epitome']));
addpath(genpath([path '\toolboxes\nx_toolbox']));
addpath([path '\toolboxes\fieldtrip']);
sep         = '\';
clearvars cwp idcs
% %% MAC
% cwp = pwd;
% idcs   = strfind(cwp,'/');
% path = cwp(1:idcs(end)-1);
% idcs   = strfind(path,'/');
% path = path(1:idcs(end)-1); 
% clearvars cwp idcs
% addpath('/Applications/MATLAB_R2020a.app/toolbox/fieldtrip');
% sep = '/';
% % addpath('/Applications/MATLAB_R2019b.app/toolbox/nx_toolbox');
% % addpath('/Applications/MATLAB_R2019b.app/toolbox/iELVis_MATLAB');
%% Get started
% addpath([pwd '/nx_plots_matlab']);
addpath([pwd '/nx_preproc']);
ft_defaults;
warning('off','MATLAB:xlswrite:AddSheet'); %optional

%% patient specific
subj            = 'EL015'; %% change name if another data is used !!
path_patient    = ['T:\EL_experiment\Patients\' subj];  
dir_files       = [path_patient,'\data_raw\EL_experiment'];
dir_infos = [path_patient,'\infos\'];
%% 1. log 
file =[dir_files '\EL016_BM_IO_1.log'];
log                 = import_logfile(file);
stimlist_all   = read_log(log);
clear log
%% load MP_label (all)
MP_label = importfile_MPlabels([dir_infos '\EL016_lookup.xlsx'], 'Channels');
BP_label = importfile_BPlabels([dir_infos '\EL016_lookup.xlsx'], 'Channels_BP');
%% 2. file
filepath               = [dir_files '/EL016_BM_IO_1.EDF']; %'/Volumes/EvM_T7/EL008/Data_raw/EL008_BM_1.edf';
H                      = Epitome_edfExtractHeader(filepath);
[hdr_edf, EEG_all]     = edfread_data(filepath);
% stimlist = stimlist_BM(stimlist_BM.type=='BM',:);
%% 3. trigger
% [hdr_edf, trig]     = edfread(filepath,'targetSignals','TRIG'); %TeOcc5, TRIG
c_trig         = find(hdr_edf.label==string('TRIG'));
trig           = EEG_all(c_trig,:);
Fs             = round(hdr_edf.frequency(1));
% [pks,locs]   = findpeaks(trig_CR1,'MinPeakDistance',2*Fs,'Threshold',0.9,'MaxPeakWidth', 0.002*Fs);
[pks,locs]     = findpeaks(trig,'MinPeakDistance',1*Fs);
locs           = locs';
ix_startblock  = find(diff(locs)/Fs>200); 
TTL_startblock = locs(ix_startblock(1)+1);
%%
stimlist.TTL = zeros(height(stimlist),1);
blocks = unique(stimlist.stim_block);
ix_block = find(stimlist.stim_block==blocks(3));
stimlist(ix_block(1),'TTL')= {TTL_startblock};
%% 4. for each stimulation, find the TTL sample
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
stimlist = stimlist(stim_noise(stim_cut(1))+1:stim_noise(stim_cut(end)+1)-1,:);
disp('TTL aligned');

%% 5. Test trigger
stimlist = stimlist_all;
clf(figure(1))
Fs     = hdr_edf.frequency(1);
%Fs = 2048;
n_trig = 50;
t      = stimlist.TTL(n_trig);
IPI    = stimlist.IPI_ms(n_trig);
x_s = 10;
x_ax        = -x_s:1/Fs:x_s;
c= 5;
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
EEG_block       = EEG_all(pos_ChanN,:)-EEG_all(pos_ChanP,:);
% % labels


%% 7.2 B / IO loop for cutting blocks
Fs = round(Fs);
type                = 'IO';

block_num               = 1;
stim_list           = stimlist(stimlist.type==type,:); 
cut_block_edf(EEG_block, stim_list,type,block_num, Fs, path_patient, subj, BP_label)

%% 7.1 CR loop for cutting blocks
type                = 'CR';
stimlist.b = stimlist.stim_block;
blocks          = unique(stimlist.b);
if height(stimlist(stimlist.b==blocks(3),:))>height(stimlist(stimlist.b==blocks(2),:))
    ix = find(stimlist.stim_block==blocks(3),1);
    stimlist.b(1:ix-2) = blocks(3);
    blocks          = unique(stimlist.b); %
end

for i=2:length(blocks)%2:11
    block_num               = blocks(i);
    stim_list           = stimlist(stimlist.b==block_num,:);%   
    cut_block_edf(EEG_block, stim_list,type,block_num, Fs, path_patient, subj, BP_label)
end
