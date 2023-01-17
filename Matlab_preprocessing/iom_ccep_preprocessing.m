%iom_ccep_preprocessing
    %script to import and preprocessing raw edf files from iom recordings
    %to process CCEPs
    %based on protocol used for IM002
    
%created RJB 6.12.22
%updated RJB 19.12.22

close all
clear all

restoredefaultpath
addpath C:\Users\rburman.MSDITUN-TMV0GCR\Dropbox\Postdoc\projects\code\CCEP_human\Matlab_preprocessing\nx_preproc

%% importing edf file
filepath                     = 'C:\Users\rburman.MSDITUN-TMV0GCR\Dropbox\Postdoc\projects\coeus\data\iom\IM002_221201\data_raw\IM002_raw_new\anonymous_2022-12-12_16-57-07\875754b7-7fb1-4861-98d5-b6e370a3aa5b\b236c547-2c90-4021-8847-29e570fdaf4c\20221201_115821_370.SEPt_08f70c4d-f0e7-45be-889d-1c2f5c776316.edf';

output_path                 = 'C:\Users\rburman.MSDITUN-TMV0GCR\Dropbox\Postdoc\projects\coeus\data\iom\IM002_221201\output\'; 

patient = 'IM002';
wave = 'alt. monophasic';
intensity = '3mA';

H                           = Epitome_edfExtractHeader(filepath);

[hdr_edf, iom_ccep_raw]     = edfread_data(filepath);

fs                          = round(hdr_edf.frequency(1)); % sampling frequency


start_time = hdr_edf.starttime;

filename = append(patient,'_',wave,'_',start_time,'_',intensity);
%% plotting raw trace per channel 

[ch, dur]               = size(iom_ccep_raw);

ch                      = ch-1; %removing the EDF annotations channel (not useful)

stim_ch                 = 1; %setting the stimulation channel (hard coded for now)

t                       = linspace(0,dur/fs,dur);

clf
figure_handle = figure(1);
set(gcf, 'Position', get(0, 'Screensize'));

for i=1:ch
   
    plot(t,iom_ccep_raw(i,:)+i*5);
                   
    ch_lab = string(hdr_edf.label(i));
        
    text(.5,i*5+1,ch_lab)
       
    ylabel('mV')
    hold on
        
end

xlabel('sec')
set(gca,'xlim',[0 10])

sgtitle('raw trace')

saveas(gcf,[output_path filename '_raw_trace_segment' '.png' ]);

%% stimulus artefact removal using EvM kriging algorithm

iom_ccep_art_removed = iom_ccep_raw;

for i=2:59 %excluding first trigger as has no preceding activity to reference for interpolaation
    
    trig = i*fs;
    
    for c=1:ch
        iom_ccep_art_removed(c,:) = kriging_artifacts_IOM(iom_ccep_art_removed(c,:), trig,trig, 0, fs,0);
    end
end

%% plotting trace after stimulus artefact rejection

clf
figure_handle = figure(2);
set(gcf, 'Position', get(0, 'Screensize'));

for i=1:ch
   
    plot(t,iom_ccep_raw(i,:)+i*5);
                   
    ch_lab = string(hdr_edf.label(i));
       
    text(.5,iom_ccep_art_removed(i,1)+i*5,ch_lab)
       
    ylabel('mV')
    hold on
        
end

xlabel('sec')

sgtitle('full raw trace')


%saveas(gcf,[output_path filename '_artefact_full_trace' '.png' ]);

%% filtering

 % bandpass 
fs2                = 20000;
[bBP, aBP]          = butter(2, [.5 200]/(fs/2), 'bandpass');
iom_ccep_filtered          = filter(bBP, aBP, iom_ccep_art_removed')';


% notch - find which notch
f_notch             = [50, 100, 150, 200];
for n=1:length(f_notch)
    fn              = f_notch(n);
    [bN, aN]        = butter(2,[fn-2 fn+2]/(fs/2),'stop');
    iom_ccep_filtered = filtfilt(bN, aN, iom_ccep_filtered')'; %filtfilt
end

iom_ccep_downsampled = resample(iom_ccep_filtered',fs2,fs)';

[ch_ds, dur_ds] = size(iom_ccep_downsampled);

t_ds = linspace(0,dur_ds/fs2,dur_ds);

%% plotting trace after filtering

clf
figure_handle = figure(3);
set(gcf, 'Position', get(0, 'Screensize'));

for i=1:ch
   
    plot(t_ds,iom_ccep_downsampled(i,:)+i*5);
                   
    ch_lab = string(hdr_edf.label(i));
        
    text(212.5,i*5+1,ch_lab)
       
    ylabel('mV')
    hold on
        
end

xlabel('sec')
set(gca,'xlim',[212.5 218.5])

sgtitle('full post filtering')

saveas(gcf,[output_path filename '_filtered_full_trace' '.png' ]);
%% cutting into epochs - raw

epoch_size = 1*fs;

epoch_number = floor(length(iom_ccep_raw(1,:))/epoch_size)-1;

iom_ccep_epoch = zeros(ch,epoch_number,epoch_size); 

for i = 1:ch
    
    for j = 1:epoch_number
    
        iom_ccep_epoch(i,j,:) = iom_ccep_raw(i,j*epoch_size-5000:(j*epoch_size-5000)+epoch_size-1);
        
    end
        
end

t_epoch = linspace(0,epoch_size/fs,epoch_size);

%% checking what the epochs of raw data look like

% for i = 2:ch
%     
%     for h = 1:epoch_number
%         
%         clf(figure(4))
%         plot(t_epoch, squeeze(iom_ccep_epoch(i,h,:)),'k')
%         pause
%                
%     end
%     
% end

%%

clf(figure(5))

tit = strcat(patient,'-',wave,'-',intensity,'-','raw');
sgtitle(tit)  
set(gcf, 'Position', get(0, 'Screensize'));

grayColor = [.7 .7 .7];
 
for i = 1:ch
    
    subplot(3,3,i)
    title(hdr_edf.label(i),'FontName', 'Arial')
    
    for h = 1:epoch_number
        
        
        plot(t_epoch,squeeze(iom_ccep_epoch(i,h,:)),'Color', grayColor)
        alpha(.25)
        title(hdr_edf.label(i),'FontName', 'Arial')
        hold on
               
    end   
    
    mean_ch_raw = mean(iom_ccep_epoch(i,:,:));
    
    subplot(3,3,i)
    plot(t_epoch,squeeze(mean_ch_raw),'k','LineWidth',2)
    
    xlabel('Time (s)'), ylabel('Voltage (mV)','FontName', 'Arial')
    set(gca,'xlim',[.2 .4])
    %set(gca,'ylim',[-.25 .25])
    set(gcf,'renderer','Painters')   
    
    xline(.25,'b','LineWidth',2)
       
end      

saveas(gcf,[output_path filename '_raw_epochs' '.png' ]);

%% cutting into epochs - processed


epoch_size = 1*fs2;

epoch_number = floor(length(iom_ccep_downsampled(1,:))/epoch_size)-1;

iom_ccep_epoch_p = zeros(ch,epoch_number,epoch_size); 

for i = 1:ch
    
    for j = 1:epoch_number
    
        iom_ccep_epoch_p(i,j,:) = iom_ccep_downsampled(i,j*epoch_size-5000:(j*epoch_size-5000)+epoch_size-1);
        
    end
        
end

t_epoch_p = linspace(0,epoch_size/fs2,epoch_size);

%% checking what the epochs of processed data look like

% for i = 2:ch
%     
%     for h = 1:epoch_number
%         
%         clf(figure(4))
%         plot(t_epoch_p, squeeze(iom_ccep_epoch_p(i,h,:)),'k')
%         xline(.25,'b','LineWidth',2)
%         %set(gca,'xlim',[.25 .30])
%         %set(gca,'ylim',[-.25 .25])         
%         pause
%                
%     end
%     
%     
% end


%% plotting epochs across channels for processed data

clf(figure(6))
tit = strcat(patient,'-',wave,'-',intensity,'-','processed');
sgtitle(tit)  
set(gcf, 'Position', get(0, 'Screensize'));

grayColor = [.7 .7 .7];
 
for i = 1:ch
    
    subplot(3,3,i)
    title(hdr_edf.label(i),'FontName', 'Arial')
    
    for h = 1:epoch_number
         
        plot(t_epoch_p,squeeze(iom_ccep_epoch_p(i,h,:)),'Color', grayColor)
        alpha(.25)
        title(hdr_edf.label(i),'FontName', 'Arial')
        hold on
               
    end   
    
    mean_ch_processed = mean(iom_ccep_epoch_p(i,:,:));
    
    subplot(3,3,i)
    plot(t_epoch_p,squeeze(mean_ch_processed),'r','LineWidth',2)
    
    xlabel('Time (s)'), ylabel('Voltage (mV)','FontName', 'Arial')
    set(gca,'xlim',[.2 .4])
    %set(gca,'ylim',[-.25 .25])
    set(gcf,'renderer','Painters')   
    
    xline(.25,'b','LineWidth',2)
    
    
end    

saveas(gcf,[output_path filename '_processed_epochs' '.png' ]);

%%

clf(figure(7))
tit = strcat(patient,'-',wave,'-',intensity,'-','means');

sgtitle(tit)  
set(gcf, 'Position', get(0, 'Screensize'));

for i = 1:ch
    
    mean_ch_raw = mean(iom_ccep_epoch(i,:,:));
    mean_ch_processed = mean(iom_ccep_epoch_p(i,:,:));  
   
    subplot(3,3,i)
    title(hdr_edf.label(i),'FontName', 'Arial')
   
    plot(t_epoch,squeeze(mean_ch_raw),'k','LineWidth',2)
    hold on
    plot(t_epoch_p,squeeze(mean_ch_processed),'r','LineWidth',2)
    
    xlabel('Time (s)'), ylabel('Voltage (mV)','FontName', 'Arial')
    set(gca,'xlim',[.2 .4])
    %set(gca,'ylim',[-.25 .25])
    set(gcf,'renderer','Painters')   
    
    xline(.25,'b','LineWidth',2)
    
    title(hdr_edf.label(i),'FontName', 'Arial')
    
end 
%%
saveas(gcf,[output_path filename '_overlaid_epochs' '.png' ]);