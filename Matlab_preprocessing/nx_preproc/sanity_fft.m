function [] = sanity_fft(EEG_raw, EEG_pp, Fs_raw, Fs_pp )
    %EEG_raw, ppEEG, Fs, Fs2, BP_label)
    x               = 6;%trig/Fs_raw;
    %labels          = table2cell(BP_label);
    ax_t_raw        = 0:1/Fs_raw:10-1/Fs_raw;
    x_raw           = x*Fs_raw;
    
    ax_t_pp         = 0:1/Fs_pp:10-1/Fs_pp;
    x_pp            = x*Fs_pp;
    
    n_img           = floor(length(EEG_raw)/8);
    n_last          = mod(length(EEG_raw),8);
%     for j=1:n_img
%         clf(figure(1))
%         if j == n_img
%             e = n_last;
%         else
%             e=8;
%         end
%         for i=1:e
%             ax(i) = subplot(e/2,2,i);
%             plot(ax_t_raw, EEG_raw((j-1)*8+i,x_raw+1:x_raw+10*Fs_raw),'Parent',ax(i))
%             hold on
%             plot(ax_t_pp, EEG_pp((j-1)*8+i,x_pp+1:x_pp+10*Fs_pp),'Parent',ax(i))
%             xlim([0,10]);
%             ylim([-300,300]);
%             title(labels((j-1)*8++i));
%             legend('raw','pp')
%         end
%         %saveas(gcf,[path, sprintf('/sanity/resp%i.png',j)])
%         %saveas(gcf,path+'/sanity/resp'+string(i)+'.png')
%     end
    %% welch
    for j=1:n_img
        clf(figure(1))
        if j == n_img
            e = n_last;
        else
            e=8;
        end
        for i=1:e
            ax(i)       = subplot(e/2,2,i);
            [pxx,f]     = pwelch(EEG_raw((j-1)*8+i,:),20*Fs_raw,[], [],Fs_raw);
            plot(f,pow2db(pxx),'Parent',ax(i))
            hold on
            [pxx,f]     = pwelch(EEG_pp((j-1)*8+i,:),20*Fs_raw,[], [],Fs_pp);
            plot(f,pow2db(pxx),'Parent',ax(i))
            

            %xlim([0, 250]);
            legend('raw','pp')
        end
        %saveas(gcf,[path, sprintf('/sanity/welch%i.png',j)])
        %saveas(gcf,path+'/sanity/resp'+string(i)+'.png')
    end


end
