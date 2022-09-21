function [pk] = LL_pk(data, w, thr, mx, IPI, Fs)
    w = 2*round(w/2);
    pads            = zeros(1,w/2);
    %data_pad        = [pads, data, pads];
    data_pad = padarray(data,[0,w/2],'replicate', 'both');
    %num             = uint16(length(data)/w);
    LL_data         = zeros(1, length(data));
    for i =1:length(LL_data)
        j = i + w/2;
        LL_data(1,i) = sum(abs(diff(data_pad(j-w/2:j+w/2))));
    end
%     plot(LL_data);
%     %yline(thr);
%     clf(figure(5))
%     ax_t        = -3:1/Fs:3;
%     plot(ax_t,LL_data);
%     xlabel('time [s]');
%     ylabel('LL');
%     hold on 
%     plot(ax_t,data);
    pk_all = find(LL_data>thr);
%     hold on 
    if IPI >0
        [~, pk_all] = findpeaks(LL_data,Fs, 'NPeaks',2,'MinPeakHeight',500, 'MinPeakDistance', IPI/1000-0.01);
        pk = round(pk_all(1)*Fs);
    %pk = pk(1);
    elseif isempty(pk_all)
        pk = find(LL_data==max(LL_data));%[];
    else
        pk = round(pk_all(1));
    end
    
    if mx
        pk = find(LL_data==max(LL_data));
    end
%     xline(pk/Fs-3);

end


 