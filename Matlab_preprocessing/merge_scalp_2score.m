clear all
close all

%% merge scalp files to score

cwp         = pwd;
idcs        = strfind(cwp,'\');
path        = cwp(1:idcs(end)-1);
idcs        = strfind(path,'\');
path        = path(1:idcs(end)-1);  % path 0, where all important folders are (Patients, codes, etc.)
%addpath('C:\Program Files\MATLAB\R2020b\toolbox\fieldtrip');
addpath(genpath([path '\elab\Epitome']));
addpath(genpath([path '\toolboxes\nx_toolbox']));
addpath([path '\toolboxes\fieldtrip']);
sep         = '\';
clearvars cwp idcs
addpath([pwd '/nx_plots_matlab']);
addpath([pwd '/nx_preproc']);
ft_defaults;
warning('off','MATLAB:xlswrite:AddSheet'); %optional
%%
subj            = 'EL015';
%block_path     = uigetdir(['E:\PhD\EL_experiment\Patients\', subj, '/Data']);
block_path     = uigetdir(['T:\EL_experiment\Patients\', subj, '/Data']); %
% block_files     = dir(block_path);
% isdir           = [block_files.isdir]; % Get all the codes
% block_files     = block_files(isdir==1); % Select only the p and H codes, delete the rest
% only CR 
block_files     = dir(block_path);
isdir           = [block_files.isdir]; % Get all the codes
block_files     = block_files(isdir==1); % Select only the p and H codes, delete the rest
%for i=3:length(block_files)
i = 3;
while i<= length(block_files)
    if  (block_files(i).name(end-3:end-2) == "BM")
        block_files(i) = [];
    elseif (block_files(i).name(end-3:end-2) == "IO")
        block_files(i) = [];
    elseif (block_files(i).name(end-3:end-2) == "Ph")
        block_files(i) = [];
    else 
        i = i+1;
    end
end
%% split based on selected files
scalp_all       = [];
score_all = [];
Fs1 = 1024;
for sf=1:height(score_files)      
    start_file = score_files.start(sf);
    stop_file = score_files.end(sf);
    for i=3:length(block_files)
        if block_files(i).name == start_file
            i_start = i;
        elseif block_files(i).name == stop_file
            i_stop = i;
        end
    end
    % concat files
    for i=i_start:i_stop
        disp(block_files(i).name);
        if i==i_start
            path = char([block_path, sep, block_files(i).name]);
            % load(char([block_files(i).folder, sep, block_files(i).name, sep,'metadata.mat']));
            [filepath,foldername]   = fileparts(path);
            [filepath]              = fileparts(filepath);
            subj                    = foldername(1:5);
            if isnan(str2double(foldername(end))) % non numeric
                type                = foldername(10:end);
                block_num           = 0;
                stim_list           =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)]);
            elseif isnan(str2double(foldername(end-1)))
                type                = foldername(10:end-1);
                block_num           = str2double(foldername(end));
                stim_list           =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
            else
                type = foldername(10:end-2);
                block_num = str2double(foldername(end-1:end));
                stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
            end
        end
        load(char([block_files(i).folder, sep, block_files(i).name, sep,'scalpEEG.mat']));
        
        % score
        try
            load(char([block_files(i).folder, sep, block_files(i).name, sep,'score.mat']));
        catch
            score = ones(1,round(length(scalpEEG)/Fs))*0;
            save(char([block_files(i).folder, sep, block_files(i).name, sep,'score.mat']),'score','-v7.3');
        end
        try
            load(char([block_path, sep, block_files(i).name, sep,'TTL.mat']));
            TTL     = round(TTL/Fs1*Fs);
        catch
            TTL = [];
        end
        
        if isempty(scalp_all)
            scalp_all = scalpEEG;
            score_all = score;
            TTL_all = TTL;
        else
            TTL     = TTL+length(scalpEEG);
            TTL_all = [TTL_all; TTL];
            scalp_all = [scalp_all, scalpEEG];
            score_all = [score_all, score];
            
        end
    end
    %scalp
    EEG = scalp_all;
    % save data
        file_name= [subj, '_scalp2score_', num2str(sf)];
        file_sel = [fileparts(block_path), sep, file_name];
         mkdir(file_sel);
        save([file_sel, sep, file_name, '.mat'],'EEG','labels', 'Fs','-v7.3');
        TTL = TTL_all;
        save([file_sel, sep, 'TTL.mat'],'TTL','-v7.3');

        score = score_all;
        save([file_sel, sep, 'score.mat'],'score','-v7.3');
        
    % metadata
    [nch,dp]    = size(EEG);
    dur         = floor(dp/Fs);
    bl          = stim_list.TTL(1)/Fs1; %metadata.fs1
    st          = sprintf('%i %i:%i:%i',stim_list.date(1),stim_list.h(1), stim_list.min(1), stim_list.s(1));
    start       = datenum(st, 'yyyymmdd HH:MM:SS')-seconds(bl);
    st          = datestr(start,'yyyymmdd HH:MM:SS');
    stop        = datenum(st, 'yyyymmdd HH:MM:SS')+seconds(dur);    

    s = dir([file_sel, sep, file_name, '.mat']);
    size_MB = max(vertcat(s.bytes))/1e6;
    create_metadata(subj,[file_sel, sep, file_name, '.mat'], Fs, nch, dp, size_MB,start,stop)

    scalp_all   = [];
    score_all   = [];
end
%% load updated score file, if not splitted
for sf=1:height(score_files)      
    start_file = score_files.start(sf);
    stop_file = score_files.end(sf);
    for i=3:length(block_files)
        if block_files(i).name == start_file
            i_start = i;
        elseif block_files(i).name == stop_file
            i_stop = i;
        end
    end
    % concat files
    %for i=i_start:i_stop
    % 1. load score file
    file_name= [subj, '_scalp2score_', num2str(sf)];
    file_sel = [fileparts(block_path), sep, file_name];
    load([file_sel, sep, 'score.mat']);
    score_all = score;
    score0 = score;
    
    for i=i_start:i_stop
        disp(block_files(i).name);
        try
            load(char([block_path, sep, block_files(i).name, sep,'score.mat']));
        catch
            %disp('no score file found')
            load(char([block_path, sep, block_files(i).name, sep,'scalpEEG.mat']));
            score = zeros(1,round(length(scalpEEG)/Fs));
        end
        score(1,:) = score_all(1, 1:size(score,2));
        save(char([block_path, sep, block_files(i).name, sep,'score.mat']),'score','-v7.3');
        
        score_all = score_all(:,size(score,2)+1:end);
    end
end
%% split files in 24h blocks
while length(block_files)>2
    
    for i=3:min(26, length(block_files))
        disp(block_files(i).name);
        if i==3
            load(char([block_path, sep, block_files(i).name, sep,'metadata.mat']));
            j=i;
        end
        load(char([block_path, sep, block_files(i).name, sep,'scalpEEG.mat']));
        if Fs>200
            disp('error');
        end
        try
            load(char([block_path, sep, block_files(i).name, sep,'TTL.mat']));
            TTL     = round(TTL/metadata.fs1*Fs);
        catch
            TTL = [];
        end

        try
            load(char([block_path, sep, block_files(i).name, sep,'score.mat']));
        catch
            %disp('no score file found')
            score = zeros(1,round(length(scalpEEG)/Fs));
        end

        if i ==3
            EEG     = scalpEEG;
            TTL_all = TTL;
            score_all = score;
        else

            TTL     = TTL+length(EEG);
            TTL_all = [TTL_all; TTL];
            EEG     = [EEG, scalpEEG];
            score_all = [score_all, score];
        end
    end
        % save data
        file_name= [subj, '_scalp2score_', num2str(k)];
        file_sel = [fileparts(block_path), sep, file_name];
         mkdir(file_sel);
        save([file_sel, sep, file_name, '.mat'],'EEG','labels', 'Fs','-v7.3');
        TTL = TTL_all;
        save([file_sel, sep, 'TTL.mat'],'TTL','-v7.3');

        score = score_all;
        save([file_sel, sep, 'score.mat'],'score','-v7.3');

        % get and save metadata

        path = char([block_path, sep, block_files(j).name]);
        [filepath,foldername]   = fileparts(path);
        [filepath]              = fileparts(filepath);
        subj                    = foldername(1:5);
        if isnan(str2double(foldername(end))) % non numeric
            type                = foldername(10:end);
            block_num           = 0;
            stim_list           =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)]);
        elseif isnan(str2double(foldername(end-1)))
            type                = foldername(10:end-1);
            block_num           = str2double(foldername(end));
            stim_list           =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
        else
            type = foldername(10:end-2);
            block_num = str2double(foldername(end-1:end));
            stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
        end

        [nch,dp]    = size(EEG);
        dur         = floor(dp/Fs);
        bl          = stim_list.TTL(1)/metadata.fs1;
        st          = sprintf('%i %i:%i:%i',stim_list.date(1),stim_list.h(1), stim_list.min(1), stim_list.s(1));
        start       = datenum(st, 'yyyymmdd HH:MM:SS')-seconds(bl);
        st          = datestr(start,'yyyymmdd HH:MM:SS');
        stop        = datenum(st, 'yyyymmdd HH:MM:SS')+seconds(dur);    

        s = dir([file_sel, sep, file_name, '.mat']);
        size_MB = max(vertcat(s.bytes))/1e6;
        create_metadata(subj,[file_sel, sep, file_name, '.mat'], Fs, nch, dp, size_MB,start,stop)
        k = k+1;
        block_files(3:min(26, length(block_files))) = [];
end
%% load updated score file, if not splitted
% load score_total
k = 1;
%file_name= [subj, '_scalp2score_', num2str(k)];
file_name= [subj, '_scalp2score'];
file_sel = [fileparts(block_path), sep, file_name];
load([file_sel, sep, 'score.mat']);
score_all = score;
score0 = score;

for i=3:length(block_files)
    disp(block_files(i).name);
    try
        load(char([block_path, sep, block_files(i).name, sep,'score.mat']));
    catch
        %disp('no score file found')
        load(char([block_path, sep, block_files(i).name, sep,'scalpEEG.mat']));
        score = zeros(1,round(length(scalpEEG)/Fs));
    end
    score(1,:) = score_all(1, 1:size(score,2));
    save(char([block_path, sep, block_files(i).name, sep,'score.mat']),'score','-v7.3');

    score_all = score_all(:,size(score,2):end);
end
%%
block_files     = dir(block_path);
isdir           = [block_files.isdir]; % Get all the codes
block_files     = block_files(isdir==1); % Select only the p and H codes, delete the rest
%for i=3:length(block_files)
i = 3;
while i<= length(block_files)
    if  (block_files(i).name(end-3:end-2) == "BM")
        block_files(i) = [];
    elseif (block_files(i).name(end-3:end-2) == "IO")
        block_files(i) = [];
    elseif (block_files(i).name(end-3:end-2) == "Ph")
        block_files(i) = [];
    else 
        i = i+1;
    end
end
%%
for i=3:length(block_files)
    score2list(char([block_path, sep, block_files(i).name]), 0);
end
%% load updated score file, if splitted
k = 1;
while length(block_files)>2
    % 1. load score file
    file_name= [subj, '_scalp2score_', num2str(k)];
    file_sel = [fileparts(block_path), sep, file_name];
    load([file_sel, sep, 'score.mat']);
    score_all = score;
    score0 = score;
    
    for i=3:min(26, length(block_files))
        disp(block_files(i).name);
        try
            load(char([block_path, sep, block_files(i).name, sep,'score.mat']));
        catch
            %disp('no score file found')
            load(char([block_path, sep, block_files(i).name, sep,'scalpEEG.mat']));
            score = zeros(1,round(length(scalpEEG)/Fs));
        end
        score(1,:) = score_all(1, 1:size(score,2));
        save(char([block_path, sep, block_files(i).name, sep,'score.mat']),'score','-v7.3');
        
        score_all = score_all(:,size(score,2):end);
    end
        
    k = k+1;
    block_files(3:min(26, length(block_files))) = [];
end
%%
subj            = 'EL011';
%block_path     = uigetdir(['E:\PhD\EL_experiment\Patients\', subj, '/Data']);
block_path     = ['T:\EL_experiment\Patients\', subj, '\Data\experiment1\data_blocks']; %
block_files     = dir(block_path);
isdir           = [block_files.isdir]; % Get all the codes
block_files     = block_files(isdir==1); % Select only the p and H codes, delete the rest

file_name= [subj, '_scalp2score'];
file_sel = [fileparts(block_path), sep, file_name];
load([file_sel, sep, 'score.mat']);
score_all = score;
score0 = score;
    
for i=3:length(block_files)
    disp(block_files(i).name);
    try
        load(char([block_path, sep, block_files(i).name, sep,'score.mat']));
    catch
        %disp('no score file found')
        load(char([block_path, sep, block_files(i).name, sep,'scalpEEG.mat']));
        score = zeros(1,round(length(scalpEEG)/Fs));
    end
    score(1,:) = score_all(1, 1:size(score,2));
    save(char([block_path, sep, block_files(i).name, sep,'score.mat']),'score','-v7.3');

    score_all = score_all(:,size(score,2):end);
end
%%
%% UPDATE STIMLIST and TTL 

%block_path     = uigetdir(['E:\PhD\EL_experiment\Patients\', subj, '/Data']);
%block_path     = uigetdir(['T:\EL_experiment\Patients\', subj, '/Data']); %
% block_files     = dir(block_path);
% isdir           = [block_files.isdir]; % Get all the codes
% block_files     = block_files(isdir==1); % Select only the p and H codes, delete the rest
for i=3:length(block_files)
    score2list(char([block_path, sep, block_files(i).name]), 0);
end