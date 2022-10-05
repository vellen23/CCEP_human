function [EEG_c] = kriging_artifacts_LT(EEG_c, trig1, trig2, IPI, Fs, scalp)
     [pk_op, pk_lp, pk_cl,pk_stim, pk_stim2] = find_artifacts_pk(EEG_c, trig1, trig2, IPI, Fs);
    
    d_pk            = round([0.005 0.005]*Fs);

    if scalp
        d_pk_stim       = round([0.005 0.015]*Fs);%[0.005 0.015]*Fs;, 2-5
    else % duration to remove
        %d_pk_stim       = round([0.005 0.015]*Fs);%[0.005 0.015]*Fs;, 2-8 , EL003:7-15
        d_pk_stim       = round([0.002 0.030]*Fs);%[0.005 0.015]*Fs;, 2-8 , EL003:7-15
    end

    le = round(0.01*Fs);%(pk_stim-d_pk_stim(1)-2) - (pk_op+d_pk(2)+1);

    %stimulations peaks
    
    EEG_c(pk_stim-d_pk_stim(1):pk_stim+d_pk_stim(2)) = kriging_func(EEG_c, pk_stim, d_pk_stim,3);
    
%     EEG_c(pk_stim-d_pk_stim(1):pk_stim+d_pk_stim(2)) = pre_mean;
    if ~isempty(pk_stim2)
        
        EEG_c(pk_stim2-d_pk_stim(1):pk_stim2+d_pk_stim(2)) = kriging_func(EEG_c, pk_stim2, d_pk_stim,3);%0;
    end
end

    