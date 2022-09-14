function [ppEEG, EEG,Fs, stim_list] = EL_preprocess(EEG_load, stim_list,sclA, sclC)
    Fs          = round(EEG_load.Fs);
    EEG_raw     = EEG_load.EEG;
    EEG         = EEG_load.EEG;
    % todo: find automatic way to load specific stim_list
    %stim_list   = EEG_load.stim_list;%eval(sprintf('stim_table_%s%i', type, block_num));%
    ax      	= (-5:1/Fs:5);
    ax_t        = -3:1/Fs:3;
    %[stim_list] = refine_TTL(EEG, stim_list, Fs);
    stim_list.TTL = stim_list.TTL+2;

    %%% 1. remove stim art, not very time efficient
    for s=1:height(stim_list) %for each stimulation
        %tic
        IPI     = stim_list.IPI_ms(s);
        trig1   = stim_list.TTL(s);
        %trig2   = stim_list.TTL_PP(s);%         
        trig2   = trig1+round(IPI/1000*Fs);
        stim_list.TTL_PP(s) = trig2;
        for c=1:size(EEG,1) %  for each channel      
            EEG(c,:) = kriging_artifacts(EEG(c,:), trig1, trig2, IPI, Fs,0);
    %         plot_raw_filter(c,s,stim_list,EEG_raw,EEG,Fs);
        end
        %toc
    end
    %%%% 2. Filter
    
    %picks = get_chs_linenoise(EEG_raw,Fs );
    % bandpass LL
    [bBP, aBP]          = butter(4, [0.5 200]/(Fs/2), 'bandpass');
    EEG_filter          = filter(bBP, aBP, EEG')';
%     % notch - find which notch
    f_notch             = [50, 100, 150, 200];
    for n=1:length(f_notch)
        fn              = f_notch(n);
        [bN, aN]        = butter(2,[fn-2 fn+2]/(Fs/2),'stop');
        EEG_filter     = filtfilt(bN, aN, EEG_filter')'; %filtfilt
    end
    EEG_scaled          = bsxfun(@rdivide,sclA*EEG_filter,sclC);
    Fs2                 = 500;
    ppEEG               = resample(EEG_scaled',Fs2,Fs)';
%     %% filter backwards (HG)
%     % bandpass LL
%     [bBP, aBP]      = butter(4, [0.5 200]/(Fs/2), 'bandpass');
%     EEG_filter      = filter(bBP, aBP, flip(EEG,2)')';
% %     % notch
%     f_notch = [50, 100, 150, 200];
%     for n=1:length(f_notch)
%         fn              = f_notch(n);
%         %[bN, aN]        = butter(2,[fn-2 fn+2]/(Fs/2),'stop');
%        [bN, aN]        = butter(2,[fn-2 fn+2]/(Fs/2),'stop');
%         EEG_filter      = filter(bN, aN, EEG_filter')'; %filtfilt
%     end
%     EEG_scaled          = bsxfun(@rdivide,sclA*EEG_filter,sclC);
%     EEG_scaled          = flip(EEG_scaled,2);
%     ppEEG_bk            = resample(EEG_scaled',Fs2,Fs)';
    
    %sanity_fft(EEG_raw, ppEEG, Fs, Fs2, BP_label)

    TTL_ds                  = round((stim_list.TTL(:))/(Fs/Fs2));
    TTL_PP_ds               = round((stim_list.TTL_PP(:))/(Fs/Fs2)); % - sample_start + 1 
    stim_list.TTL_DS(:)     = TTL_ds;
    stim_list.TTL_PP_DS(:)  = TTL_PP_ds;
    Fs                      = Fs2;    
end
