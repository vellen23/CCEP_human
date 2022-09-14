function save_metadata(path)
    [filepath,foldername]   = fileparts(path);
    [filepath]              = fileparts(filepath);
    subj                    = foldername(1:5);
    if isnan(str2double(foldername(end))) % non numeric
        type            = foldername(10:end);
        block_num       = 0;
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)]);
    elseif isnan(str2double(foldername(end-1)))
        type            = foldername(10:end-1);
        block_num       = str2double(foldername(end));
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
    else
        type            = foldername(10:end-2);
        block_num       = str2double(foldername(end-1:end));
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
    end
    EEG_load = load([path, '/' foldername '.mat']);
    Fs          = EEG_load.Fs;
    EEG         = EEG_load.EEG;
    %meta data
    [nch,dp]    = size(EEG);
    bl          = floor(stim_list.TTL(1)/Fs);
    dur         = floor(dp/Fs);
    if ~any(startsWith(stim_list.Properties.VariableNames, 'date'))
        d = zeros(height(stim_list),1);
        d = d+20210126;%d+input('date? yyyymmdd:    ');
        stim_list.date = d;
        if block_num >0
            writetable(stim_list,[filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
        else
            writetable(stim_list,[filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)]);
        end
    end    
    st          = sprintf('%i %i:%i:%i',stim_list.date(1),stim_list.h(1), stim_list.min(1), stim_list.s(1));
    start       = datenum(st, 'yyyymmdd HH:MM:SS')-seconds(bl);
    st          = datestr(start,'yyyymmdd HH:MM:SS');
    stop        = datenum(st, 'yyyymmdd HH:MM:SS')+seconds(dur);
    s           = dir([path, '/' foldername '.mat']);
    size_MB = max(vertcat(s.bytes))/1e6;
    create_metadata(subj,[path, '/' foldername '.mat'], Fs, nch, dp, size_MB,start,stop)

end