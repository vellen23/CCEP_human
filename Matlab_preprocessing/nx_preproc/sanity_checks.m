function [] = sanity_checks(path,BP_label )
    % load infos
    % score2list scores the stimulations from the stim_list based on the
    % Epitome scoring 
    %path = 'T:\EL_experiment\Patients\EL005\Data\experiment1\data_blocks\EL005_BP_CR1';
    BP_label = BP_label(contains(BP_label.type,'SEEG'),:);
    [filepath,foldername]   = fileparts(path);
    [filepath]              = fileparts(filepath);
    subj                    = foldername(1:5);
    if isnan(str2double(foldername(end))) % non numeric
        type = foldername(10:end);
        block_num = 0;
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)]);
        EEG_load                = load([path, sprintf('/%s_BP_%s.mat',subj, type)]);
    elseif isnan(str2double(foldername(end-1)))
        type = foldername(10:end-1);
        block_num = str2double(foldername(end));
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
        EEG_load                = load([path, sprintf('/%s_BP_%s%i.mat',subj, type, block_num)]);
    else
        type = foldername(10:end-2);
        block_num = str2double(foldername(end-1:end));
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type)],'Sheet',block_num);
        EEG_load                = load([path, sprintf('/%s_BP_%s%i.mat',subj, type, block_num)]);
    end
    %
    mkdir([path, '/sanity/'])
    
    EEGpp_load              = load([path, '/ppEEG.mat']);
    
    Fs_raw     = EEG_load.Fs;
    EEG_raw    = EEG_load.EEG;
    Fs_pp      = EEGpp_load.fs;
    EEG_pp     = EEGpp_load.ppEEG;
    
    trig            =  stim_list.TTL(10);
    x               = 6;%trig/Fs_raw;
    labels          = table2cell(BP_label);
    ax_t_raw        = 0:1/Fs_raw:10-1/Fs_raw;
    x_raw           = x*Fs_raw;
    
    ax_t_pp         = 0:1/Fs_pp:10-1/Fs_pp;
    x_pp            = x*Fs_pp;
    
    n_img           = floor(length(labels)/8);
    n_last          = mod(length(labels),8);
    for j=1:n_img
        clf(figure(1))
        figg = figure('visible','off');
        if j == n_img
            e = n_last;
        else
            e=8;
        end
        for i=1:e
            ax(i) = subplot(4,2,i);
            plot(ax_t_raw, EEG_raw((j-1)*8+i,x_raw+1:x_raw+10*Fs_raw),'Parent',ax(i))
            hold on
            plot(ax_t_pp, EEG_pp((j-1)*8+i,x_pp+1:x_pp+10*Fs_pp),'Parent',ax(i))
            xlim([0,10]);
            ylim([-300,300]);
            title(labels((j-1)*8++i));
            %legend('raw','pp')
        end
        saveas(figg,[path, sprintf('/sanity/resp%i.png',j)])
        %saveas(gcf,path+'/sanity/resp'+string(i)+'.png')
    end
    %% welch
    for j=1:n_img
        clf(figure(1))
        figg = figure('visible','off');
        if j == n_img
            e = n_last;
        else
            e=8;
        end
        for i=1:e
            ax(i)       = subplot(4,2,i);
            [pxx,f]     = pwelch(EEG_raw((j-1)*8+i,:),20*Fs_raw,[], [],Fs_raw);
            plot(f,pow2db(pxx),'Parent',ax(i))
            hold on
            [pxx,f]     = pwelch(EEG_pp((j-1)*8+i,:),20*Fs_raw,[], [],Fs_pp);
            plot(f,pow2db(pxx),'Parent',ax(i))
            
            title(labels((j-1)*8++i));
            xlim([0, 250]);
            %legend('raw','pp')
            ylabel('PSD (dB/Hz)')
            xlabel('freq')
        end
        saveas(figg,[path, sprintf('/sanity/welch%i.png',j)])
        %saveas(gcf,path+'/sanity/resp'+string(i)+'.png')
    end


end
