function [EEG_c] = kriging_artifacts(EEG_c, trig1, trig2, IPI, Fs, scalp)
     [pk_op, pk_lp, pk_cl,pk_stim, pk_stim2] = find_artifacts_pk(EEG_c, trig1, trig2, IPI, Fs);
    
    d_pk            = round([0.005 0.005]*Fs);
    d_pk_l          = round([0.002 0.002]*Fs);
    d_pk_l          = round([0.002 0.009]*Fs);
    d_pk_c          = round([0.003 0.003]*Fs);
    if scalp
        d_pk_stim       = round([0.005 0.015]*Fs);%[0.005 0.015]*Fs;, 2-5
    else
        %d_pk_stim       = round([0.005 0.015]*Fs);%[0.005 0.015]*Fs;, 2-8 , EL003:7-15
        d_pk_stim       = round([0.002 0.022]*Fs);%[0.005 0.015]*Fs;, 2-8 , EL003:7-15
    end

    le = round(0.01*Fs);%(pk_stim-d_pk_stim(1)-2) - (pk_op+d_pk(2)+1);
    RMS_pre = nanmedian(EEG_c(pk_op-d_pk(1)-le:pk_op-d_pk(1)));
    RMS_DC  = nanmedian(EEG_c(pk_op+d_pk(2)+1:pk_stim-d_pk_stim(1)-2));
%     yline(RMS_DC, '--','LinesWidth',1.2);
%     yline(RMS_pre, '--','LineWidth',1.2);
    % DC shiftssss
    %if RMS_DC-RMS_pre>-std(EEG_c(pk_op-d_pk(1)-l:pk_op-d_pk(1)))
%     if abs(RMS_DC-RMS_pre)>1.5*std(EEG_c(pk_op-d_pk(1)-round(0.21*Fs):pk_op-d_pk(1)))
%         pre_mean        = mean(EEG_c(pk_op-d_pk(1)-le:pk_op-d_pk(1)));
%         %pre_mean        = mean(EEG_c(pk_op-50:pk_op-d_pk(1)-2));
%         %post_mean       = median(EEG_c(trig2+20:trig2+200));%mean(EEG_c(pk_stim+d_pk_stim(2):pk_stim+d_pk_stim(2)+10));
%         DC_start        = EEG_c(pk_op+d_pk(2)+1)-pre_mean;
%         DC_end          = EEG_c(trig1-d_pk_stim(1))-pre_mean;
%         dur             = (trig1-d_pk_stim(1))-(pk_op+d_pk(2)+1);
%         if DC_end<DC_start
%             DC_shift_lin    = flip(linspace(DC_end,DC_start,dur));%flip(DC_end:(abs(DC_start-DC_end))/dur:DC_start);
%         else
%             DC_shift_lin    = linspace(DC_start,DC_end,dur);%DC_start:(abs(DC_end-DC_start))/dur:DC_end;
%         end
%         if ~isempty(DC_shift_lin)
%            EEG_c(pk_op+d_pk(2)+1:pk_op+d_pk(2)+dur) = EEG_c(pk_op+d_pk(2)+1:pk_op+d_pk(2)+dur)-DC_shift_lin;
%         end
%     end
    % artifact peaks
    EEG_c(pk_op-d_pk(1):pk_op+d_pk(2)) = kriging_func(EEG_c, pk_op, d_pk,3);
    if ~isempty(pk_lp)
        EEG_c(pk_lp-d_pk_l(1):pk_lp+d_pk_l(2)) = kriging_func(EEG_c, pk_lp, d_pk_l,3);
    end
    
    if ~isempty(pk_cl)
        EEG_c(pk_cl-d_pk_c(1):pk_cl+d_pk_c(2)) = kriging_func(EEG_c, pk_cl, d_pk_c,2);
    end
    

    %stimulations peaks
    
    EEG_c(pk_stim-d_pk_stim(1):pk_stim+d_pk_stim(2)) = kriging_func(EEG_c, pk_stim, d_pk_stim,3);
    
%     EEG_c(pk_stim-d_pk_stim(1):pk_stim+d_pk_stim(2)) = pre_mean;
    if ~isempty(pk_stim2)
        
        EEG_c(pk_stim2-d_pk_stim(1):pk_stim2+d_pk_stim(2)) = kriging_func(EEG_c, pk_stim2, d_pk_stim,3);%0;
    end
end

    