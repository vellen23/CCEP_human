function [] = rerun_pp(path, sclA, sclC )
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
    % PREPROCESS EEG
    if block_num>0
        EEG_load                = load([path, sprintf('/%s_BP_%s%02d.mat',subj, type, block_num)]);
%         c                       = parcluster('Castor');
%         job = batch(c, @EL_preprocess_castor, 5, {EEG_load, stim_list,sclA, sclC}, 'CurrentDirectory', pwd, 'Pool', 24);
%         wait(job);
%         diary(job);
%         results         = fetchOutputs(job);
%         ppEEG           = results{1,1};
%         ppEEG_bk        = results{1,2};
%         EEG_art         = results{1,3};
%         fs              = results{1,4};
%         stim_list       = results{1,5};
        [ppEEG,EEG_art,fs, stim_list]       = EL_preprocess(EEG_load, stim_list,sclA, sclC);
        % save ppEEG and new stimlist
        writetable(stim_list,[filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);

    else
        EEG_load                    = load([path, sprintf('/%s_BP_%s.mat',subj, type)]);
        [ppEEG,EEG_art,fs, stim_list]      = EL_preprocess(EEG_load, stim_list,sclA, sclC);
        % save ppEEG and new stimlist
        
        writetable(stim_list,[filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)]);
    end
    save([path, '/ppEEG.mat'],'ppEEG', 'fs','-v7.3');
%     ppEEG = ppEEG_bk;
%     save([path, '/ppEEG_bk.mat'],'ppEEG', 'fs','-v7.3');
    Fs          = round(EEG_load.Fs);
    save([path, '/EEG_art.mat'],'EEG_art', 'Fs','-v7.3');

    %     % PREPROCESS SCALP
    %     labels                  = cellstr(BP_label.label(contains(BP_label.type,'scalp')));
    %     EEG_load                = load([path, '/scalpEEG.mat']);
    %     [scalpEEG, Fs,~]        = EL_preprocess_scalp(EEG_load,stim_list,sclA_scalp, sclC_scalp,labels);
    %     save([path, '/scalpEEG.mat'],'scalpEEG','Fs', 'labels','-v7.3');
end
