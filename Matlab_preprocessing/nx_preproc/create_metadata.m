function create_metadata(subj, filepath, fs, nch, dp, size_MB,start,stop)
        % print
        % get folder name
        [path,folder,~]        = fileparts(filepath);
        ln                  = 50;
        dur                 = (dp/fs)/60; 
        fid                 = fopen([path,'/Metadata.txt'],'w'); % print link to source file in directory
        fprintf(fid,'Last open: %s \n',datestr(now,'dd-mmm-yyyy'));
        fprintf(fid,['Subject ID: ' subj '\n']);
        fprintf(fid,['Processing folder: ' folder '\n']); % folder name
        fprintf(fid, 'Raw data Source: %s\n', filepath)%fprintf(fid,['Raw data Source: ' filepath '\n']);
        fprintf(fid,'File size: %.1f MB\n',size_MB);
        fprintf(fid,'Datapoints:  %d dp\n',dp);
        fprintf(fid,'Duration:  %.1f min\n',dur);
        fprintf(fid,'File starts: %s\n',datestr(start,'dd-mmm-yyyy HH:MM:SS'));
        fprintf(fid,'File stops: %s\n', datestr(stop, 'dd-mmm-yyyy HH:MM:SS'));
        fprintf(fid,'Number of channels: %d\n',nch);
        fprintf(fid,'RawEEG fs: %d Hz \n',fs);
        fprintf(fid,'Line-noise: %d Hz \n',ln);
        fclose(fid);
        disp('metadata stored');
    end