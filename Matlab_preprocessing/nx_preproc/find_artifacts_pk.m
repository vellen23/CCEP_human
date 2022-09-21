function [pk_op, pk_lp, pk_cl,pk_stim, pk_stim2] = find_artifacts_pk(EEG_c, trig1, trig2, IPI, Fs)
    %find peak of SM opening artifact, ~160ms before stim
    dp = round(0.005*Fs);
    
    SM_0                = round(trig1 - 0.250*Fs);
    SM_1                = round(trig1 - 0.09*Fs); %140
    pk                  = LL_pk(EEG_c(SM_0:SM_1), dp, 0, true,0,Fs);
    %[~,locs]            = max(abs(EEG_c(SM_0:SM_1)-mean(EEG_c(SM_0:SM_1))));%findpeaks(EEG_c(SM_0:SM_1),'MinPeakProminence',2*pre_std);
    pk_op               = round(pk(1)+SM_0); 
%     if isempty(pk_op)
%         pk_op = trig1 - 318;
%     end

    % little peak, ~25ms before stim
    SM_0                = round(trig1 - 0.03*Fs);
    SM_1                = round(trig1 - 0.017*Fs);
    locs                = LL_pk(EEG_c(SM_0:SM_1), dp, 0, true,0,Fs);
    %[~,locs]            = max(abs(EEG_c(SM_0:SM_1)-mean(EEG_c(SM_0:SM_1))));%findpeaks(abs(EEG_c(SM_0:SM_1)),'MinPeakProminence',pre_std,'NPeaks',1);
    pk_lp               = round(locs + SM_0);

    % 
    % SM closing, ~300ms after stim
    dp = round(0.003*Fs);
    SM_0                = round(trig2 + 0.240*Fs);
    SM_1                = round(trig2 + 0.500*Fs);
    if SM_1<length(EEG_c)
        locs                = LL_pk(EEG_c(SM_0:SM_1), dp, 200, true,0,Fs);
    %[~,locs]            = max(abs(EEG_c(SM_0:SM_1)-mean(EEG_c(SM_0:SM_1))));%findpeaks(abs(EEG_c(SM_0:SM_1)),'MinPeakProminence',pre_std,'NPeaks',1);
        pk_cl               = round(locs + SM_0);
    else
        pk_cl               = round(trig2 + 0.278*Fs);
    end
    if length(pk_cl)>1
    pk_cl=pk_cl(1);
    if max(abs(EEG_c(1,pk_cl-0.001*Fs:pk_cl+0.001*Fs)))<mean(abs(EEG_c(1,pk_cl+0.003*Fs:pk_cl+0.006*Fs)))+std(abs(EEG_c(1,pk_cl+0.003*Fs:pk_cl+0.006*Fs)))
        pk_cl=[];
    end
    end
    % stim peak, INTERPOLATION METHOD, KRIGING (SUP)
%     [~,locs]            = max(abs(EEG_c(trig1:trig1+0.007*Fs)-median(EEG_c(trig1:trig1+0.007*Fs))));%findpeaks(EEG_c(SM_0:SM_1),'MinPeakProminence',2*pre_std);
%     pk_stim             = locs+trig1; 
    pk_stim             = round(trig1);
%     [~,locs]        = findpeaks(abs(EEG_c(trig1:trig1+0.005*Fs)),'MinPeakHeight',thr_stim,'NPeaks',1);
%     pk_stim         = locs+trig1;
    if IPI  > 0
%         [~,locs]            = max(abs(EEG_c(trig2:trig2+0.007*Fs)-median(EEG_c(trig2:trig2+0.007*Fs))));%findpeaks(EEG_c(SM_0:SM_1),'MinPeakProminence',2*pre_std);
%         pk_stim2             = locs+trig2;
        pk_stim2             = round(trig2);
%         [~,locs]            = findpeaks(abs(EEG_c(trig2:trig2+0.005*Fs)),'MinPeakHeight',thr_stim,'NPeaks',1);
%         pk_stim2            = locs+trig2; 
    else
        pk_stim2 =[];
    end
end

