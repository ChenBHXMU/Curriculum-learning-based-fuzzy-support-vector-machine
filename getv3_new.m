function [d] = getv3_new(M, para)
    % getv3_new 计算 d，其中 d(i) = 2/(1 + exp(para * M(i)))
    % M - 输入矩阵或向量
    % para - 参数值，默认为 0.5
    
    if nargin < 2 || para == 0
        para = 0.5;  % 如果 para 没有指定或为 0，默认为 0.5
    end
    
    % 矢量化计算，直接对整个矩阵/向量 M 进行操作
    d = 2 ./ (1 + exp(para * M(:)));
    % d = d';
end

