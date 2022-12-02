
function [] = plot_raw_filter(c,s,stim_list,EEG_raw,EEG_scaled,Fs)
    clf(figure(1))
    trig1       = stim_list.TTL(s);
    thr         = 1*Fs;
    w_0         = trig1 - thr;
    w_1         = trig1 + 3*thr;
    n_sample    = w_1-w_0;
    timeX_s     = (0:1/Fs:n_sample/Fs)-thr/Fs;
    %%
    [bBP, aBP]          = butter(4, [0.0005 200]/(Fs/2), 'bandpass');
    EEG_filter          = filter(bBP, aBP, EEG_scaled(c, w_0:w_1)')';
%     % notch - find which notch
    f_notch             = [50, 100, 150, 200];
    for n=1:length(f_notch)
        fn              = f_notch(n);
        [bN, aN]        = butter(2,[fn-2 fn+2]/(Fs/2),'stop');
        EEG_filter     = filtfilt(bN, aN, EEG_filter')'; %filtfilt
    end
    %% 
    plot(timeX_s,EEG_raw(c, w_0:w_1))
    hold on
    plot(timeX_s,EEG_scaled(c, w_0:w_1))
     hold on
    plot(timeX_s,EEG_filter)
    IPI = stim_list.IPI_ms(s);
    Int = stim_list.Int_prob(s);
    xlabel('time [s]','FontSize',25)
    ylabel('uV','FontSize',25)
%     if IPI > 0
%         title(BP_label.label(c)+' IPI: '  + string(IPI)+'ms, Int: '+string(stim_list.Int_cond(s))+'-'+string(Int)+'mA','FontSize',27)
%     else
%         title(BP_label.label(c)+', Int: '+string(Int)+'mA','FontSize',27)
%     end
    xline(0, '--','LineWidth',1.2);
    xline(IPI/1000, '--','LineWidth',1.2);
    yline(0, '--','LineWidth',1.2);
    ylim([-100,100]);
    xlim([-1,2]);
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',20,'FontWeight','bold')
    set(gca,'XTickLabelMode','auto')
    legend('raw', 'art rem', 'filter ', 'FontSize', 20)
    
end

    