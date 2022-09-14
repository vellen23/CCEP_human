function [scalpEEG, scalpFS,scalpTTL] = EL_preprocess_scalp(EEG_load,stim_list,sclA, sclC)
% remove stim artifacts (not as good as in icEEG), downsample and filter
% data
% store new data as the same as before (FS to be read in epitome)
% todo: find automatic way to load specific stim_list
    %stim_list   = EEG_load.stim_list;%eval(sprintf('stim_table_%s%i', type, block_num));%

    %[stim_list] = refine_TTL(EEG, stim_list, Fs);
    scalpFS     = round(EEG_load.scalpFs);
    EEG_raw     = EEG_load.scalpEEG;
    EEG_raw =EEG_raw(1:length(sclC),:);
    EEG         = EEG_load.scalpEEG;
    EEG =EEG(1:length(sclC),:);
    ax      	= (-5:1/scalpFS:5);
    ax_t        = -3:1/scalpFS:3;
    %%% 1. remove stim art
%     for s=1:height(stim_list) %for each stimulation
%         IPI     = stim_list.IPI_ms(s);
%         trig1   = stim_list.TTL(s);
%         %trig2   = stim_list.TTL_PP(s);%         
%         trig2   = trig1+round(IPI/1000*scalpFS);
%         stim_list.TTL_PP(s) = trig2;
%         for c=1:size(EEG,1) %  for each channel      
%             EEG(c,:) = kriging_artifacts(EEG(c,:), trig1, trig2, IPI, scalpFS, 1);
%     %         plot_raw_filter(c,s,stim_list,EEG_raw,EEG,scalpFS);
%         end
%     end
    %%%% 2. Filter
    % bandpass
    [bBP, aBP]              = butter(2, [0.5 80]/(scalpFS/2), 'bandpass');
    EEG                     = filter(bBP, aBP, EEG')';
%     % notch
    
    f_notch             = [50, 100, 150, 200];
    for n=1:length(f_notch)
        fn              = f_notch(n);
        [bN, aN]        = butter(2,[fn-2 fn+2]/(scalpFS/2),'stop');
        EEG     = filtfilt(bN, aN, EEG')'; %filtfilt
    end
    

    EEG                     = bsxfun(@rdivide,sclA*EEG,sclC);
    Fs2                     = 200;
    scalpEEG                = resample(EEG',Fs2,scalpFS)';
    scalpTTL_ds             = round((stim_list.TTL(:))/(scalpFS/Fs2));
    scalpTTL_PP_ds          = round((stim_list.TTL_PP(:))/(scalpFS/Fs2)); %-sample_start+1
    scalpFS                 = Fs2;    
    scalpTTL                = [scalpTTL_ds,scalpTTL_PP_ds];

end