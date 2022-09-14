function [index] = find_BP_index(labels, ChanP, ChanN)
    index       = zeros(length(ChanP),2);
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