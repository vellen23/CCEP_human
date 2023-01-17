%%
%iom_ccep_preprocessing
    %script to import and preprocessing raw edf files from iom recordings
    %to process CCEPs
    %based on protocol used for IM002
    
%created RJB 7.12.22
%updated RJB 20.12.22 

close all
clear all

restoredefaultpath
addpath C:\Users\rburman.MSDITUN-TMV0GCR\Dropbox\Postdoc\projects\code\CCEP_human\Matlab_preprocessing\nx_preproc

%% importing edf file

%for TEST001
filepath                    = 'C:\Users\rburman.MSDITUN-TMV0GCR\Dropbox\Postdoc\projects\coeus\data\iom\TEST001\data_raw\anonymous_2022-12-05_15-09-57\3da601e4-87c5-4184-8c04-999959599774\6450df79-082e-41c5-8314-b6f7faa397e9\20221122_133317_746.SEPt_557abfce-d708-44db-a646-c7946a7bdac4.edf';

output_path                 = 'C:\Users\rburman.MSDITUN-TMV0GCR\Dropbox\Postdoc\projects\coeus\data\iom\TEST001\output\'; 


H                           = Epitome_edfExtractHeader(filepath);

[hdr_edf, iom_ccep_raw]     = edfread_data(filepath);

fs                          = round(hdr_edf.frequency(1)); % sampling frequency

%% plotting raw trace per channel 

[ch, dur]               = size(iom_ccep_raw);

ch                      = ch-1; %removing the EDF annotations channel (not useful)

stim_ch                 = 1; %setting the stimulation channel (hard coded for now)

t                       = linspace(0,dur/fs,dur);

figure_handle = figure(1);

for i=1:ch
   
    subplot(ch,1,i)
    plot(t,iom_ccep_raw(i,:),'b','LineWidth',2);
                
    title(hdr_edf.label(i),'FontName', 'Arial')
    
    if i <ch
        
        set(gca,'XTick',[]);
        
    end
    ylabel('mV','FontName', 'Arial')
        
end

sgtitle('TEST001 - 221222 13:33:17 (Bipolar, 1.1Hz, PW 500us, 20mA, single pulse ISI 40ms) asleep ')
xlabel('Time (s)')
all_ha = findobj( figure_handle, 'type', 'axes', 'tag', '' );
linkaxes( all_ha, 'x' );

%%
clf
figure_handle = figure(2);
%title(hdr_edf.label(i),'FontName', 'Arial')

for i = 1:ch
    
    plot(t,iom_ccep_raw(i,:)+i,'LineWidth',2);
    ylabel('mV','FontName', 'Arial')           
    
    ch_lab = string(hdr_edf.label(i));
        
    text(.05,i,ch_lab)    


    xlabel('Time (s)')
    %set(gca,'xlim',[0 .1])
    %set(gca,'ylim',[0 5.5])
    hold on
end

saveas(gcf,[output_path 'TEST001_CCEP_133317_asleep' '.png' ]);

%%
figure_handle = figure(3);
plot(t,iom_ccep_raw(15,:)*1000,'b','LineWidth',1);
set(gca,'xlim',[0 .1])
%set(gca,'ylim',[-.2 .2])
title(hdr_edf.label(15),'FontName', 'Arial')
xlabel('Time (s)')
ylabel('Voltage (uV)')

saveas(gcf,[output_path 'TEST001_CCEP_133317_single_34_asleep' '.png' ]);