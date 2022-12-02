function [] = cut_block_edf(EEG_block, stim_list,type,block_num, Fs, subj,BP_label, path_pp)
    % function cuts edf file into block based on stimulations (from stim_list) and stores it in a way that Epitome is able to ed it.     
    % labels:
    inSEEG      = find(contains(string(BP_label.type),'SEEG')|contains(string(BP_label.type),'GRID')|contains(string(BP_label.type),'STRIP'));
    inScalp     = find(contains(string(BP_label.type),'scalp'));
    inECG       = find(contains(string(BP_label.type),'EKG')|contains(string(BP_label.type),'ecg'));

    labels_all      = table2cell(BP_label);
    % cuts block into -5min before first stim and +6s after last stim
    start_sample    = max(1,stim_list.TTL(1)-300*Fs);
    end_sample      = min(size(EEG_block,2),stim_list.TTL(end)+6*Fs);
    stim_list.TTL       = stim_list.TTL-start_sample+1;
    blocklength     = (end_sample-start_sample)/(Fs*60); % in minutes
    fprintf('%s %s. block: %s min\n', type, string(block_num), string(blocklength));
    % save EEG blockj
    EEG         = EEG_block(inSEEG, start_sample:end_sample);
    scalpEEG    = EEG_block(inScalp, start_sample:end_sample);
    ECG         = EEG_block(inECG, start_sample:end_sample);
    scalpFs     = Fs;
    % metadata
    [nch,dp]    = size(EEG);
    dur         = floor(dp/Fs);
    bl          = stim_list.TTL(1)/Fs;
    st          = sprintf('%i %i:%i:%i',stim_list.date(1),stim_list.h(1), stim_list.min(1), stim_list.s(1));
    start       = datenum(st, 'yyyymmdd HH:MM:SS')-seconds(bl);
    st          = datestr(start,'yyyymmdd HH:MM:SS');
    stop        = datenum(st, 'yyyymmdd HH:MM:SS')+seconds(dur);
    % store data
    mkdir(sprintf('%s/data_blocks/%s_BP_%s%02d',path_pp,subj, type, block_num))
    labels = labels_all(inSEEG,1);
     %save([path, sprintf('/data_blocks/time/%s_BP_%s_block_%i.mat',subj, type, block_num)],'EEG','labels', 'Fs','start_sample','-v7.3');
    save(sprintf('%s/data_blocks/%s_BP_%s%02d/%s_BP_%s%02d.mat',path_pp,subj, type, block_num,subj, type, block_num),'EEG','labels', 'Fs','-v7.3');
    labels = labels_all(inScalp,1);
    save(sprintf('%s/data_blocks/%s_BP_%s%02d/scalpEEG.mat',path_pp,subj, type, block_num),'scalpEEG','scalpFs','labels','-v7.3');
    if ~isempty(inECG)
        EMG    = EEG_block(inECG,:);
        EMG_label = labels_all(inECG);
        save(sprintf('%s/data_blocks/%s_BP_%s%02d/EMG.mat',path_pp,subj, type, block_num),'EMG','EMG_label', 'Fs','-v7.3');
    end
       
    s = dir(sprintf('%s/data_blocks/%s_BP_%s%02d/%s_BP_%s%02d.mat',path_pp,subj, type, block_num,subj, type, block_num));
    size_MB = max(vertcat(s.bytes))/1e6;
    create_metadata(subj,sprintf('%s/data_blocks/%s_BP_%s%02d/%s_BP_%s%02d.mat',path_pp,subj, type, block_num,subj, type, block_num), Fs, nch, dp, size_MB,start,stop)
    % stim_list = removevars(stim_list, 'b');
    writetable(stim_list,sprintf('%s/%s_stimlist_%s.xlsx',path_pp,subj, type),'Sheet',block_num);            
    fprintf('Data Saved -- /%s_BP_%s_%02d\n',subj, type, block_num);
end
