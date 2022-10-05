function [] = cut_block_edf_stimlistONLY(EEG_block, stim_list,type,block_num, Fs, subj,BP_label, path_pp)
    % cuts block into -5min before first stim and +6s after last stim
    start_sample    = max(1,stim_list.TTL(1)-300*Fs);
    end_sample      = stim_list.TTL(end)+6*Fs;% min(size(EEG_block,2),stim_list.TTL(end)+6*Fs);
    stim_list.TTL       = stim_list.TTL-start_sample+1;
    blocklength     = (end_sample-start_sample)/(Fs*60); % in minutes
    fprintf('%s %s. block: %s min\n', type, string(block_num), string(blocklength));
    
    writetable(stim_list,sprintf('%s/%s_stimlist_%s.xlsx',path_pp,subj, type),'Sheet',block_num);            

end
