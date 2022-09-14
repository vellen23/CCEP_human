function [stim_table_all] = read_log(log)
    % without date, from first imp check
    log.Properties.VariableNames{6} = 'type';
    log.Properties.VariableNames{2} = 'h';
    log.Properties.VariableNames{1} = 'date';
    log.Properties.VariableNames{3} = 'min';
    log.Properties.VariableNames{4} = 's';
    log.Properties.VariableNames{5} = 'us';
    
    log.Properties.VariableNames{7} = 'StimNum';
    log.Properties.VariableNames{8} = 'ChanP';
    log.Properties.VariableNames{9} = 'ChanN';
    log.Properties.VariableNames{10} = 'ISI_s';
    log.Properties.VariableNames{12} = 'Int_cond';
    log.Properties.VariableNames{11} = 'Int_prob';
    log.Properties.VariableNames{13} = 'IPI_ms';
    log.Properties.VariableNames{14} = 'stim_block';
    log.Properties.VariableNames{15} = 'currentflow';
    stim_table_all= log(log.type~='Comment' &log.type~='SwitchMatrix' & log.type~='Baseline'& log.type~='Start'& log.type~=' SM Connected'& log.type~='Stim Connected',:);
    %stim_table_all = log(log.type=='BM' |log.type=='IO'|log.type=='CR'|log.type=='PP'|log.type=='AD'|log.type=='Ph',:);
    %stim_table_all = log(log.type=='CR_IO'|log.type=='CR_PP',:);
%     stim_table_all = log(log.type=='Ph_IO'|log.type=='Ph_PP'|log.type=='Ph_BM'|log.type=='BM' ,:);
%     SM_table = log(log.type=='SwitchMatrix',:);
%     SM_table.Properties.VariableNames{6} = 'State';Stim Connected
%     SM_table.Properties.VariableNames{7} = 'ChanP';
%     SM_table.Properties.VariableNames{8} = 'ChanN';
end
%% pharmacology tables 
% stim_table_all = log(log.type=='Pharmacology',:);
% stim_table_IO_1 = stim_table_all(stim_table_all.stim_block==0 & stim_table_all.IPI_ms==0,:);
% stim_table_IO_2 = stim_table_all(stim_table_all.stim_block==1 & stim_table_all.IPI_ms==0,:);
% stim_table_IO_3 = stim_table_all(stim_table_all.stim_block==2 & stim_table_all.IPI_ms==0,:);
% stim_table_CR_1 = stim_table_all(stim_table_all.stim_block==0 & stim_table_all.IPI_ms>0,:);
% stim_table_CR_2 = stim_table_all(stim_table_all.stim_block==1 & stim_table_all.IPI_ms>0,:);
% stim_table_CR_3 = stim_table_all(stim_table_all.stim_block==2 & stim_table_all.IPI_ms>0,:);