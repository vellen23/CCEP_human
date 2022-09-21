function [pk] = LL_pk_stim(data, w, Fs, IPI,t)
        %t = 3;
        w = 2*w;
%         pads            = zeros(1,w/2);
%         data_pad        = [pads, data, pads];
        data_pad = padarray(data,[0,w/2], 'symmetric', 'both');
        %num             = uint16(length(data)/w);
        LL_data         = zeros(size(data));

        for i =1:length(LL_data)
            j = i + w/2;
            LL_data(:,i) = sum(abs(diff(data_pad(:,j-w/2:j+w/2),1,2)),2);
        end
        LL_data = median(LL_data,1);
       
    if IPI ==0
        pks = zeros(1,2);
        for k =1:2
            pks(1,k) = find(LL_data==max(LL_data),1); % probably 1. stimulation artifact
            if pks(1,k)<round(0.01*Fs)
                LL_data(1,1: round(pks(1,k)+0.006*Fs)) = 0; % blank out
            elseif pks(1,k)>round(length(data)-0.006*Fs)
                LL_data(1,round(pks(1,k)-0.010*Fs): end) = 0; % blank out
            else
                LL_data(1,round(pks(1,k)-0.003*Fs): round(pks(1,k)+0.003*Fs)) = 0; % blank out
            end   
%             pks(1,2) = find(LL_data==max(LL_data),1);% probably 2. stimulation artifact
%             LL_data(1,round(pks(1,2)-0.010*Fs): round(pks(1,2)+0.010*Fs)) = 0; % blank out
%             pks(1,3) = find(LL_data==max(LL_data),1);% probably SM opening artifact
        end
%         pks(1,1) = find(LL_data==max(LL_data),1); % probably stimulation artifact
%         LL_data(1,pks(1)-2*w: pks(1)+2*w) = 0; % blank out
%         pks(1,2) = find(LL_data==max(LL_data),1);% probably SM opening artifact

        %pks = sort(pks(1,:));
        pk = pks(1,1); %1,2
        clf(figure(3))
        ax_t        = -t:1/Fs:t;
        if length(LL_data) ~= length(ax_t)
            disp('error');
        end
        plot(ax_t,LL_data);
        xlabel('time [s]');
        ylabel('LL');
        xline(pk/Fs-t, '-r','LineWidth',2);
        xline( pks(1,1)/Fs-t);
        hold on
        plot(ax_t,mean(data,1));
        xlim([-0.2,0.2]);
    else
%         [~, pk_all] = findpeaks(LL_data,Fs, 'NPeaks',2,'MinPeakHeight',700, 'MinPeakDistance', IPI/1000-0.01);
%         pk      = round(pk_all(1)*Fs);
%         IPI_test = round((pk_all(2)-pk_all(1))/Fs*1000);
        %pk = round(pk_all(1)*Fs);
        pks = zeros(1,3);
        for k =1:3
            pks(1,k) = find(LL_data==max(LL_data),1); % probably 1. stimulation artifact
            if pks(1,k)<round(0.01*Fs)
                LL_data(1,1: round(pks(1,k)+0.010*Fs)) = 0; % blank out
            else
                LL_data(1,round(pks(1,k)-0.010*Fs): round(pks(1,k)+0.010*Fs)) = 0; % blank out
            end   
%             pks(1,2) = find(LL_data==max(LL_data),1);% probably 2. stimulation artifact
%             LL_data(1,round(pks(1,2)-0.010*Fs): round(pks(1,2)+0.010*Fs)) = 0; % blank out
%             pks(1,3) = find(LL_data==max(LL_data),1);% probably SM opening artifact
        end
        LL_data =LL_data(1,1:size(data,2));
        pks = sort(pks(1,:));
        pk = pks(1,2);
        IPI_test = round((pks(1,3)-pks(1,2))/Fs*1000);
        if abs(IPI - IPI_test)>5
            pk = pks(1,1);
            IPI_test = round((pks(1,2)-pks(1,1))/Fs*1000);
            IPI_test2 = round((pks(1,3)-pks(1,1))/Fs*1000);
            if abs(IPI - IPI_test)>5 && abs(IPI - IPI_test2)>5
                clf(figure(3))
                ax_t        = -t:1/Fs:t;
                plot(ax_t,LL_data);
                xlabel('time [s]');
                ylabel('LL');
                xline(pk/Fs-t, '-r');
                xline( pks(1,2)/Fs-t);
                xline( pks(1,3)/Fs-t);
                hold on
                plot(ax_t,mean(data,1));
            end
        end    
    end


    %%
%     clf(figure(3))
%     ax_t        = -3:1/Fs:3;
%     plot(ax_t,LL_data);
%     xlabel('time [s]');
%     ylabel('LL');
%     xline(pk/Fs-3);
%     hold on
%     plot(ax_t,mean(data,1));


    
end
