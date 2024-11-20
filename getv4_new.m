function [d] = getv4_new(M, para)
    % getv4_new 计算 d，其中 d(i) = exp(-((M(i) - para)^2 / 2))
    % M - 输入矩阵或向量
    % para - 参数值，默认为 0
    
    % if nargin < 2
    %     para = 0;  % 如果 para 没有指定，默认为 0
    % end
    para = 0;
    para2 = 2;  % 固定参数 para2 为 2
    
    % 矢量化计算，直接对整个矩阵/向量 M 进行操作
    d = exp(-((M(:) - para).^2) / para2);
    % d = d';
end
