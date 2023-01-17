%% 1. some random data and LL transform with different LL window
Fs = 10; % sampling freq = 10 Hz
dur = 100; % 100s long data
x = 1:dur*Fs;
ax_s     = 0:1/Fs:dur-(1/Fs);
A =cos(2*pi*0.05*x+2*pi*rand) + 0.5*randn(1,dur*Fs);
A(1, 300:500) = A(1, 300:500) + 10+abs(randn(1,201));
B = sin(3*pi*0.1*x+2*pi*rand) + 1.5*randn(1,dur*Fs);
data = [A;B];
clf(figure(1))
LL_1s = get_linelength(data, Fs, 20);
LL_2s = get_linelength(data, Fs, 2);
LL_10s = get_linelength(data, Fs, 10);
plot(ax_s,data(1,:))
hold on
plot(ax_s,LL_1s(1,:))
hold on
plot(ax_s,LL_2s(1,:))
hold on
plot(ax_s,LL_10s(1,:))
legend('Raw', 'LL [0.5s]', 'LL [2s]', 'LL [10s]');

%% 2. find a threshold for baseline. E.g. 99th percentile 
% select prefered LL transform
LL = LL_1s;
BL_end = 20; % define window of BL: e.g. 1-30s of data
thr = prctile(LL(:,1:BL_end*Fs), 95,2);
clf(figure(2))
plot(ax_s,data(1,:))
hold on
plot(ax_s,LL(1,:))
hold on
yline(thr(1))
% todo:
% find a way to detect when LL crosses treshold. add half window size to
% get to the real timepoint  when data crosses threshold
%% 
function [LLall] = get_linelength(data, Fs, w_s)
    [nch,t]     = size(data); % number of channels/ neurons / cells and duration (in timepoints)
    wdp           = ceil((w_s*Fs)/2)*2; % LL window in samplepoints
    
    data_pad = padarray(data, [0 wdp/2],'replicate','both'); % pad half window size in second dimension  
    % To optimize computation, Loop over 40000 datapoints
    LLall = zeros(size(data)); % LL has the same size as input data
    for i=1:t 
        n = i + (wdp / 2);
        LLall(:,i)= nansum(abs(diff( data_pad(:,n-(wdp/2):n+(wdp/2)),1,2)),2); % get rid of pad
     
    end
end

