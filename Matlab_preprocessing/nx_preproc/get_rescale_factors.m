function [sclA, sclC] = get_rescale_factors(EEG_data, Fs, scalp, bl)
    % bl: how many seconds are used for baseline to caclulate scaling factors
    % fig: true if figure should be printed
    % bl normally 5 min, 300s
    %labels          = table2cell(BP_label);
    EEG             = EEG_data(:,1:bl*Fs);
    if scalp
        %[bBP, aBP]      = butter(2, [0.5 80]/(Fs/2), 'bandpass');
        [bBP, aBP]      = butter(2, [4 30]/(Fs/2), 'bandpass');
        EEG             = filter(bBP, aBP, EEG')';
    %     % notch
        f_notch             = [50, 100];
        for n=1:length(f_notch)
            fn              = f_notch(n);
            [bN, aN]        = butter(2,[fn-2 fn+2]/(Fs/2),'stop');
            EEG      = filtfilt(bN, aN, EEG')'; %filtfilt
        end

    else

        [bBP, aBP]      = butter(4, [4 30]/(Fs/2), 'bandpass');
        EEG             = filter(bBP, aBP, EEG')';
        f_notch             = [50, 100, 150, 200, 250];
        for n=1:length(f_notch)
            fn              = f_notch(n);
            [bN, aN]        = butter(2,[fn-2 fn+2]/(Fs/2),'stop');
            EEG      = filtfilt(bN, aN, EEG')'; %filtfilt
        end
    end
    
    A           = abs(EEG(:,:)); %baseline, normally 5min
    % pseudobackground
    B           = nanmean(A);
    B           = B < nanmedian(B);
    A           = A(:,B);
%     % scale
    scl         = nanmean(A,2);     % mean of each channel
    sclA        = 20/nanmean(scl);%nanmean(scl)/50; %20/nanmean(scl); % across EEGs normalization, set background amplitude to 50 µV
    sclC        = scl/nanmean(scl); % Sc
    EEG_scaled = bsxfun(@rdivide,sclA*EEG,sclC);
    ax_t    = 0:1/Fs:10-1/Fs;
    
    
%     x = 20*Fs;%100*Fs;%;
%     lead = 0;
% 
%         figure(1)
%         for i=1:size(EEG_data,1)/10
%             ax(2*i-1) = subplot(10,2,2*i-1);
%             plot(ax_t, EEG(lead+i,x+1:x+10*Fs),'Parent',ax(2*i-1))
%             xlim([0,10]);
%             ylim([-300,300]);
% %             title(labels(lead+i));
% 
%             ax(2*i) = subplot(10,2,2*i);
%             plot(ax_t, EEG_scaled(lead+i,x+1:x+10*Fs),'Parent',ax(2*i))
%             xlim([0,10]);
%             ylim([-300,300]);
% %             title(labels(lead+i));
%             linkaxes(ax,'xy');
% 
%         end
% 
%     sgtitle('Before and after Rescaling')
end
