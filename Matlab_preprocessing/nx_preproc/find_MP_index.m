function [index] = find_MP_index(labels, ChanP)
    index       = zeros(length(ChanP),1);
    for i=1:length(ChanP)
        k = find(labels==string(ChanP(i)));
        if ~isempty(k)
            index(i,1)= k;
        else
            index(i,1)= length(labels)+1;
        end
    end 
end