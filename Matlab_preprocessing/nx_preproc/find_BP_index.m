function [index] = find_BP_index(labels, ChanP, ChanN)
    index       = zeros(length(ChanP),2);
    if iscell(labels)
        for i=1:length(ChanP)
            disp(ChanP(i));
            disp(ChanN(i));
            index(i,1)= find(strcmp(labels, string(ChanP(i)))==1);
            index(i,2)= find(strcmp(labels, string(ChanN(i)))==1);
        end 
    else
        for i=1:length(ChanP)
            k = find(labels==string(ChanP(i)));
            if ~isempty(k)
                index(i,1)= k;
            else
                index(i,1)= length(labels)+1;
            end

            k = find(labels==string(ChanN(i)));
            if ~isempty(k)
                index(i,2)= k;
            else
                index(i,2)= length(labels)+1;
            end
        end 
    end
end