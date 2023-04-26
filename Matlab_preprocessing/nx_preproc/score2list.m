function [] = score2list(path, badtimes)

    % score2list scores the stimulations from the stim_list based on the
    % Epitome scoring 
    %path = 'T:\EL_experiment\Patients\EL005\Data\experiment1\data_blocks\EL005_BP_CR1';
    [filepath,foldername] = fileparts(path);
    [filepath] = fileparts(filepath);
    subj = foldername(1:5);
    if isnan(str2double(foldername(end))) % non numeric
        type = foldername(10:end);
        type_excel = type;
        block_num = 0;
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type_excel)],'NumHeaderLines',0);
    elseif isnan(str2double(foldername(end-1)))
        type = foldername(10:end-1);
        block_num = str2double(foldername(end));
        type_excel = type;
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type_excel)],'NumHeaderLines',0,'Sheet',block_num);
    else
        type = foldername(10:end-2);
        if type(end)=="_"
            type_excel = type(1:end-1);
        else
            type_excel = type;
        end
        block_num = str2double(foldername(end-1:end));
        stim_list       =  readtable([filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type_excel)],'NumHeaderLines',0,'Sheet',block_num);
    end
    Fs = load([path, '/' foldername '.mat'], 'Fs');
    Fs = Fs.Fs;
    %% sleep score
    try
        load([path, '/score.mat']);
    catch
        %disp('no score file found')
        score = zeros(1,12000);
    end
    
    
    % sleep score

    Fs_score    = 1;

    score_fs    = repelem(score, Fs);
    
    x = 0:9;
    [tf, loc] = ismember(score_fs,x);
    %[x, ~, indAval] = unique(scores);
    %wake, n1,n2,n3,rem, benzo, fluma, .., .., sz
    Avalnew = [0,1,2,3,4,0,9,0,0,0];
    score_fs = Avalnew(loc);
    
    stim_list.sleep = zeros(size(stim_list,1), 1);    %0
    stim_list.sz = zeros(size(stim_list,1), 1);    %0= BL, 1: preictal, 2: ictal, 3: postical
    if contains(type,'CR')
        stim_list.condition = zeros(size(stim_list,1), 1); % CR = 0, BL =1, Flu =2, Benzo = 3
    elseif contains(type,'Ph')
        stim_list.condition = ones(size(stim_list,1), 1)*block_num;
    else
        stim_list.condition = zeros(size(stim_list,1), 1); % CR = 0, BL =1, Flu =2, Benzo = 3
    end
    
    sz_ix = find(score_fs==9);
    
        
    %scores = zeros(length(EEG),1);
    for i=1:height(stim_list)
        stim_list.sleep(i) = score_fs(round(stim_list.TTL(i)));
        if size(sz_ix,2)>0
            if stim_list.TTL(i)<sz_ix(1) - 3600*Fs
                stim_list.sz(i) =0;
            elseif stim_list.TTL(i)<sz_ix(1)
                stim_list.sz(i) =1;
            elseif stim_list.TTL(i)<sz_ix(end)
                stim_list.sz(i) =2;
            elseif stim_list.TTL(i)<sz_ix(end)+ 3600*Fs
                stim_list.sz(i) =3;end
        end
           
    end
    if badtimes
        %% bad times, noise
        %% bad times
        %bad_t = Q.bad_times;
        try
            load([path, '/bad_times.mat']);
            bad_t =bad_times;
        catch
            %disp('no bad times')
            bad_t = zeros(1,Fs*40000);
        end
            % bad times from sleep score (8= artifacts)
            score_fs    = repelem(score, Fs);
            % stim_list.noise(:) = 0;
            %scores = zeros(length(EEG),1);
            for i=1:height(stim_list)
                if score_fs(round(stim_list.TTL(i)))==8
                    stim_list.noise(i) = 1;
                end
            end

            bad_t = bad_t*Fs;
            for i=1:height(stim_list)
                for t=1:size(bad_t,1)
                    if stim_list.TTL(i)>bad_t(t,1)&&stim_list.TTL(i)<bad_t(t,2)
                        stim_list.noise(i) = 1;
                    end
                end
            end
        
    end
    if block_num>0
        writetable(stim_list,[filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type_excel)],'Sheet',block_num);
    else
        writetable(stim_list,[filepath, sprintf('/%s_stimlist_%s.xlsx',subj, type_excel)]);
    end
    disp('data saved')
end
