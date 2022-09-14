function [TTL_data,TTL_table] = fix_TTL_nx(dir_files)
%% updated read neuralynx (new protocols)

% nx recording ignores gaps to keep continous data, however, the TTL sample point is still including the gaps. --> substract the amount of sample point during gap from TTL samples points
% 	1. calculated  TTL sample point as if there was no gaps. (timestamp - timestamp(1))/500
% 	2. get the gaps
% 
% 		1. 
% --> load timestamp data of one channel ->  ft_read_data([dir_files '/Amy02.ncs'], 'timestamp', true);
% 		2. 
%  get differences and look for diff>500 (gaps) and <500 (duplicates)
% 	3. 
% get gaps and duplicates durations (timestamp of datapoint after gap - timestamp of datapoint before gap)
% 	4. 
% go to TTL_table (read_event). substract (resp. add) the gap (dupl) durations from TTL sample point that are greater than the data point of the gap (dupl)

    hdr_data    = ft_read_header(dir_files); % Open header
    chan_data   = ft_read_data(dir_files);   % Open data
    TTL_data        = ft_read_event(dir_files);  % Open event TTL
    %% 1. find gaps
    orig            = ft_read_header([dir_files '/CA01.ncs']);
    Fs              = orig.Fs;
    % get timestamp of each sample point of recording.
    ts          = ft_read_data([dir_files '/CA01.ncs'], 'timestamp', true);
%time difference
%dt = (length(ts)-length(chan_data))/Fs;

    mn              = ts(1); %first timestamp
    mx              = ts(end); % last timestamp
    ts1             = ts(1);%first timestamp
    md              = mode(diff(double(ts-ts1)));
%%
    dt_us           = 1000000/Fs; % ideal time difference between sample points (microseconds/fs)
    mode_dts        = mode(md);         %500 (500us per sample)
    rng             = double(mx-mn);    % total time in microseconds
    tsinterp        = [0:mode_dts:rng]; % real timestamp axis (without breaks)
    dt              = (length(tsinterp)-length(chan_data))/Fs; %how many seconds time difference? e.g. 60s time differences
%
    tss             = double(ts-ts1); %orignal timestamp, but starting at 0
%tss    = round(tss/500)*500;
    ts_diff         = diff(tss); %ideally filled with 500 (if Fs = 2000)
    gaps            = find(diff(tss)>dt_us+1); % sample points where next sample is not 500us away -> gap
    dupl            = find(diff(tss)<dt_us-1); % duplicates - sample ploints where next sample points appear before

    gaps_dur        = ts_diff(gaps);
    gaps_dur_sam    = gaps_dur/500; 
    dupl_dur        = ts_diff(dupl);
    dupl_dur_sam    = dupl_dur/500;
%   %create corr table with sample point uf gap/dups and duration of
%   gap/dups in sample ploints
    %dt2 = (sum(gaps_dur_sam)+sum(dupl_dur_sam))/Fs; %should be the same as dt

%%
    ttl_cont = [[TTL_data.sample].',[TTL_data.sample].',[TTL_data.timestamp].',[TTL_data.sample].'];
    %1. original sample, 2. sample based on timestamp, 4. sample resolved by gaps and duplicates
    ttl_cont(:,2) = (ttl_cont(:,3)-ttl_cont(1,3))/dt_us+1;
    ttl_cont(:,4) = ttl_cont(:,2);

%% TODO find solution
for i =1:height(corr)
    disp(corr.sample(i));
    ttl_cont(ttl_cont(:,2)>corr.sample(i),4) = ttl_cont(ttl_cont(:,2)>corr.sample(i),4)-round(corr.dur(i));
end
% add last colum to ttl_table as TTL
TTL_data.TTL = ttl_cont(:,4);
%%
TTL_sample      = [];
TTL_timestamp   = [];
for i=1:length(TTL_data)
%     if TTL_data(i).value == 1
    if isequal({TTL_data(i).string} ,{'TTL'})
        TTL_sample      = [TTL_sample;TTL_data(i).sample];%TTL
        TTL_timestamp   = [TTL_timestamp;TTL_data(i).timestamp];
    end
end

TTL_table  = table(1,2);
TTL_table.Properties.VariableNames{1} = 'sample';
TTL_table.Properties.VariableNames{2} = 'PP';
TTL_table.TTL(1) = TTL_sample(1);
TTL_table.timestamp(1) = TTL_timestamp(1);
TTL_table.PP(1) = 0;
for i=1:length(TTL_sample)-1
    TTL_sample_dif          = TTL_sample(i+1) - TTL_sample(i); %  TTL_sample(i+1) - TTL_sample(i);
    TTL_table.TTL(i+1)      = TTL_sample(i+1);
    TTL_table.timestamp(i+1) = TTL_timestamp(i+1);
    if TTL_sample_dif < 2*Fs % 1s
        TTL_table.PP(i+1) = TTL_table.PP(i)+1; 
    else
        TTL_table.PP(i+1) = 0;
    end
end
end
