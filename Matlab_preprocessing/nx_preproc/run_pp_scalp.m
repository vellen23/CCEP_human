function [] = rerun_pp_scalp(path, sclA_scalp, sclC_scalp, BP_label )
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
        if type(end)=="_"
            type_excel = type(1:end-1);
        else
            type_excel = type;
        end
        block_num = str2double(foldername(end-1:end));
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type_excel)],'Sheet',block_num);
    end
    % PREPROCESS EEG

    EEG_load                 = load([path, '\scalpEEG.mat']);
    if isfield(EEG_load,'scalpFs')
        if EEG_load.scalpFs>200
            labels                   = cellstr(BP_label.label(contains(string(BP_label.type),'scalp')));%cellstr(BP_label.label(contains(BP_label.type,'scalp')));
            [scalpEEG, Fs, ~] = EL_preprocess_scalp(EEG_load,stim_list,sclA_scalp, sclC_scalp);
            %save([path_patient, sprintf('/Data/experiment%i/data_blocks/%s_BP_%s%i/scalpEEG.mat',exp,subj, type,b)],'scalpEEG','Fs','scalpTTL', 'labels','-v7.3');

            save([path, '/scalpEEG.mat'],'scalpEEG', 'Fs', 'labels','-v7.3');
        end
    end
end
