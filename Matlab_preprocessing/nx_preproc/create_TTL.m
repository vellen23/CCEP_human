function [] = create_TTL(path)

    % score2list scores the stimulations from the stim_list based on the
    % Epitome scoring 
    %path = 'T:\EL_experiment\Patients\EL005\Data\experiment1\data_blocks\EL005_BP_CR1';
    [filepath,foldername] = fileparts(path);
    [filepath] = fileparts(filepath);
    subj = foldername(1:5);
    if isnan(str2double(foldername(end))) % non numeric
        type = foldername(10:end);
        block_num = 0;
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)]);
    elseif isnan(str2double(foldername(end-1)))
        type = foldername(10:end-1);
        block_num = str2double(foldername(end));
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
    else
        type = foldername(10:end-2);
        block_num = str2double(foldername(end-1:end));
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
    end
    TTL1 = stim_list.TTL;
    TTL = stim_list.TTL_PP;
    TTL = unique(sort(cat(1, TTL1, TTL)));
    save([path, '/TTL.mat'], 'TTL');
    disp('data saved')
end
