function [d] = getv2_new(M, pN, nN)
    % 计算 f=1-1/max(xi)
    epsilon = 1e-5;
    
    % 计算正负样本部分的最大值并加上 epsilon
    maxpd = max(M(1:pN)) + epsilon;
    maxnd = max(M(pN+1:end)) + epsilon;
    
    % 初始化 d
    d = ones(length(M),1);
    
    % 对正样本和负样本分别计算 d
    d(1:pN,1) = 1 - M(1:pN) / maxpd;
    d(pN+1:end,1) = 1 - M(pN+1:end) / maxnd;
end
