
%% Windows, load toolboxes
cwp         = pwd;
sep         = '\';
idcs        = strfind(cwp,sep);
path        = 'T:\EL_experiment\Codes';
%addpath([path '\toolboxes\fieldtrip']);
idcs        = strfind(path,sep);
path        = path(1:idcs(end)-1);  % path 0, where all important folders are (Patients, codes, etc.)
%addpath('C:\Program Files\MATLAB\R2020b\toolbox\fieldtrip');
% addpath(genpath([path '\elab\Epitome']));
addpath(genpath([path '\toolboxes\nx_toolbox']));

clearvars cwp idcs
addpath([pwd '/nx_preproc']);
% ft_defaults;
warning('off','MATLAB:xlswrite:AddSheet'); %optional
%% 1. define protocl and folders
path = 'X:\\4 e-Lab\\Patients\\';
subj            = 'EL028'; %% change name if another data is used !!
path_patient    = [path,  subj];  
dir_files       = [path_patient,'/data_raw/Pharmaco'];
prot_type                = 'Ph';
path_pp = [path_patient '\Data\Pharmaco\']; 
files_raw = dir([dir_files '\*' prot_type '*.EDF']);
%% 2. load log file 

log_files = dir([dir_files '\*' prot_type '*.log']);

% Initialize an empty table to hold all concatenated data
all_stimlist = table();

for log_ix = 1:length(log_files)
    log             = importfile_log_2([dir_files '\' log_files(log_ix).name]);
    % Convert the first element of the categorical array to string and then to double
    num_date = str2double(string(log.date(1)));

    % Filter out rows where log.date is "WAIT"
    stimlist_single = log(log.date ~= "WAIT", :);
    % Filter rows where the "type" column starts with "Ph"
    % Remove rows where the "type" column is NaN
    stimlist_single = stimlist_single(~isundefined(stimlist_single.type), :);
    % Convert 'type' column to string for startsWith function
    type_str = string(stimlist_single.type);

    % Filter rows where the "type" column starts with "Ph"
    stimlist_single = stimlist_single(startsWith(type_str, 'Ph'), :);

    % Update column names
    stimlist_single.Properties.VariableNames{8} = 'stim_block';
    stimlist_single.Properties.VariableNames{2} = 'h';
    stimlist_single.keep = ones(height(stimlist_single), 1);

    % Convert categorical dates to numeric
    stimlist_single.date = str2double(string(stimlist_single.date));

    % Calculate the corresponding dates for each row
    correspondingDates = datetime(num_date, 'ConvertFrom', 'yyyymmdd') + days(stimlist_single.date - num_date);

    % Convert datetime objects back to integer date values in the format yyyymmdd
    integerDates = year(correspondingDates) * 10000 + month(correspondingDates) * 100 + day(correspondingDates);

    % Update the "date" column in the table with integer date values
    stimlist_single.date = integerDates;
    % Concatenate the current stimlist_all to the all_stimlist table
    stimlist_all = [stimlist_all; stimlist_single];
end

%%
for j=2:length(files_raw)
    %% 1. read first raw data
    file = files_raw(j).name;
    filepath               = [dir_files '/' file]; %'/Volumes/EvM_T7/EL008/Data_raw/EL008_BM_1.edf';
    H                      = Epitome_edfExtractHeader(filepath);
    [hdr_edf, EEG_all]     = edfread_data(filepath);
    stimlist = stimlist_all;
    stimlist = removevars(stimlist, 'keep');
    %% 2. find triggers
    % [hdr_edf, trig]     = edfread(filepath,'targetSignals','TRIG'); %TeOcc5, TRIG
    c_trig         = find(hdr_edf.label=="TRIG"); % find(hdr_edf.label=="EDFAnnotations"); %
    trig           = EEG_all(c_trig,:);
    Fs             = round(hdr_edf.frequency(1));
    % [pks,locs]   = findpeaks(trig_CR1,'MinPeakDistance',2*Fs,'Threshold',0.9,'MaxPeakWidth', 0.002*Fs);
    [pks,locs]     = findpeaks(trig,'MinPeakDistance',1*Fs);
    locs           = locs';
    ix_startblock  = find(diff(locs)/Fs>180); 
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
    stimlist_all.noise = stimlist.noise;
    stimlist_all.TTL = stimlist.TTL;
    % Find indices where the 'noise' values switches from 1 -->0
    start_indices = find(diff([0; stimlist_all.noise]) == -1);

    % Find indices where the 'noise' values start to be 1 again 0 --> 1
    end_indices = find(diff([stimlist_all.noise; 0]) == 1);

    % Check if the 'noise' values already start with  0s
    if stimlist_all.noise(2) == 0
        start_indices = [1; start_indices];
    end

    % Check if the 'noise' values end with 0  
    if stimlist_all.noise(end) == 0
        end_indices = [end_indices; numel(stimlist_all.noise)];
    end

    % Check if there are at least two consecutive 1s at the start
    if numel(start_indices) >= 2 && start_indices(2) - start_indices(1) == 1
        start_indices = start_indices(2:end);
    end

    % Check if there are at least two consecutive 1s at the end
    if numel(end_indices) >= 2 && end_indices(end) - end_indices(end-1) == 1
        end_indices = end_indices(1:end-1);
    end

    % Extract the smaller table 'stimlist'
    if ~isempty(start_indices) && ~isempty(end_indices)
        start_index = start_indices(1);
        end_index = end_indices(end);

        stimlist = stimlist(start_index:end_index, :);
        stimlist_all = stimlist_all(end_index+1:end, :);
    else
        disp('some technical problems');
    end

    % Display the smaller table
    disp('TTL aligned');
    stimlist_all= removevars(stimlist_all,{'noise', 'TTL'});
%% Warning if misaignemnet was bad
A = stimlist.noise;
a=cumsum(A)+1;
if a(end)>0.1*length(a)
    disp('check trigger alignment again');
end

    %% 5. Test trigger
    clf(figure(1))
    Fs     = hdr_edf.frequency(1);
    %Fs = 148;
    n_trig = 29;
    t      = stimlist.TTL(n_trig);
    IPI    = stimlist.IPI_ms(n_trig);
    x_s = 10;
    x_ax        = -x_s:1/Fs:x_s;
    c= 5;% stimlist.ChanP(n_trig);
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
    
    %% 7.1 Ph loop for cutting blocks
    stimlist = stimlist(stimlist.type~="CR_triplet",:);
    stimlist.b = stimlist.stim_block;
    blocks          = unique(stimlist.b);
    blocks = blocks(blocks>0);
    for i=1:length(blocks)%2:11
        block_num               = blocks(i);
        stim_list           = stimlist(stimlist.b==block_num,:);%   
        cut_block_edf(EEG_all, stim_list,prot_type,block_num, Fs, subj, BP_label,path_pp)
        % cut_block_edf(EEG_block, stim_list,type,block_num, Fs, subj,BP_label, path_pp)
        %(EEG_block, stim_list,type,block_num, Fs, subj,BP_label, path_pp)
    end
    assignin('base',['stimlist_' file(1:end-4)], stimlist)

end

