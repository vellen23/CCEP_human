function rescale(path, sclA, sclC)
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
    
    
%     EEG_load    = load([path, '/ppEEG0.mat']);
%     fs          = EEG_load.fs;
% %     labels      = EEG_load.labels;
%     EEG         = EEG_load.ppEEG;
%     ppEEG          = bsxfun(@rdivide,sclA*EEG,sclC);
%     save([path,'/ppEEG0.mat'],'ppEEG', 'fs','-v7.3');
    EEG_load    = load([path, '/ppEEG.mat']);
    fs          = EEG_load.fs;
    EEG         = EEG_load.ppEEG;
    ppEEG          = bsxfun(@rdivide,sclA*EEG,sclC);
    save([path,'/ppEEG.mat'],'ppEEG', 'fs','-v7.3');
    
    
end