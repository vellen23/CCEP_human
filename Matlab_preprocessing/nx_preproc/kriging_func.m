function [kriging_lin] = kriging_func(EEG_c, pk, dur_blank,f)
    dur         = sum(dur_blank)+1;
    pk = round(pk(1));
    dur_blank = round(dur_blank);
    pre_median  = median(EEG_c(round(pk-dur_blank(1)-5):round(pk-dur_blank(1))));
    pre_std     = std(EEG_c(round(pk-dur_blank(1)-dur-10):round(pk-dur_blank(1)-10)));
    post_median = median(EEG_c(round(pk+dur_blank(2)):round(pk+dur_blank(2)+3)));%median(EEG_c(pk+dur_blank(2):pk+dur_blank(2)+dur));
    
    if post_median<pre_median
        kriging_lin    = flip(linspace(post_median,pre_median,dur));
        %kriging_lin = flip(post_median:abs(post_median-pre_median)/dur:pre_median);
    else
        kriging_lin    = linspace(pre_median,post_median,dur);
        %kriging_lin = pre_median:abs(post_median-pre_median)/dur:post_median;
    end
    if ~isempty(kriging_lin)
            kriging_lin = kriging_lin+random('norm', 0, pre_std/f, [1,length(kriging_lin)]);
    else
        kriging_lin = random('norm', 0, pre_std, [1,length(kriging_lin)]);
    end
    if length(kriging_lin) ~= length(pk-dur_blank(1):pk+dur_blank(2))
        disp(length(pk-dur_blank(1):pk+dur_blank(2)));
        kriging_lin = EEG_c(pk-dur_blank(1):pk+dur_blank(2));
    end
    
end

    