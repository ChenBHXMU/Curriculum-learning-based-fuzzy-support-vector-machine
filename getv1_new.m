function [d] = getv1_new(M)
    % d = ones(length(M),1);
    % 计算f=1/(1+|xi|)
    d = 1 ./ (1 + abs(M)); % 直接计算 d，矢量化操作
    d = d';
end

