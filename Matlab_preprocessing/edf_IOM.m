%% edf file
filepath               = 'C:\Users\rburman.MSDITUN-TMV0GCR\Dropbox\Postdoc\projects\coeus\data\anonymous_2022-09-23_14-37-36\ccep_protocol\20220923_103625_827.SEPt_62d44fa8-c7f7-4614-8b07-9008a50e6ff8.edf';
H                      = Epitome_edfExtractHeader(filepath);
[hdr_edf, EEG_SEP]     = edfread_data(filepath);
%% plots
Fs             = round(hdr_edf.frequency(1)); % recording Fs

c = 1;
t      = 2000000;
x_s = 10;
%x_ax        = -x_s:1/Fs:x_s;
x_ax        = 1:1/Fs:20*60;

%% stimulus artefact removal
Fs             = round(hdr_edf.frequency(1)); % recording Fs

EEG_pp = EEG_SEP;

[h,w] = size(EEG_SEP);

for i=1:1200
    trig = i*Fs;
    
    for c=1:h
        EEG_pp(c,:) = kriging_artifacts_IOM(EEG_pp(c,:), trig,trig, 0, Fs,0);
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


%%
clf(figure(1))
plot(EEG_SEP(c,:),'b');
hold on 
plot(EEG_pp(c,:),'r');
hold on 
plot(EEG_filter(c,:),'g');

title(hdr_edf.label(c))
