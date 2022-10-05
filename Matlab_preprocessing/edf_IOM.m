%% edf file
filepath = 'F:\raw data anonimized\anonymous_2022-09-23_14-37-36\c0685166-ffa7-46a1-aa73-d8b302eb910c\387d08a6-6e0c-41df-a1ad-2b24ccade91e\20220923_103625_827.SEPt_62d44fa8-c7f7-4614-8b07-9008a50e6ff8.edf';
H                      = Epitome_edfExtractHeader(filepath);
[hdr_edf, EEG_SEP]     = edfread_data(filepath);
%% plots
Fs             = round(hdr_edf.frequency(1)); % recording Fs
clf(figure(1))

c = 1;
t      = 2000000;
x_s = 10;
%x_ax        = -x_s:1/Fs:x_s;
x_ax        = 1:1/Fs:20*60;
plot(EEG_SEP(c,:));
hold on 
plot(EEG_pp(c,:));
hold on 
plot(EEG_filter(c,:));
title(hdr_edf.label(c))
%%

EEG_pp = EEG_SEP;
for i=1:1200
    trig = i*Fs;
    for c=1:height(EEG_SEP)
        EEG_pp(c,:) = kriging_artifacts_LT(EEG_pp(c,:), trig,trig, 0, Fs,0);
    end
end

%% filtering
 % bandpass LL
 Fs2                = 500;
[bBP, aBP]          = butter(2, [2 200]/(Fs/2), 'bandpass');
EEG_filter          = filter(bBP, aBP, EEG_pp')';
%     % notch - find which notch
f_notch             = [50, 100, 150, 200];
for n=1:length(f_notch)
    fn              = f_notch(n);
    [bN, aN]        = butter(2,[fn-2 fn+2]/(Fs/2),'stop');
    EEG_filter     = filtfilt(bN, aN, EEG_filter')'; %filtfilt
end
% EEG_scaled          = bsxfun(@rdivide,sclA*EEG_filter,sclC);

% ppEEG               = resample(EEG_scaled',Fs2,Fs)';