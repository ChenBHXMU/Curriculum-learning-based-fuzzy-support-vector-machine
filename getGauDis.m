function [distance] = getGauDis(X)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%X= X';
[~,N] = size(X);
distance = zeros(N,N);
X2 = sum( X.^2 , 1 );
distance = repmat( X2 , N , 1 ) + repmat( X2' , 1 , N ) - 2 * X' * X;  % 2范数距离矩阵
d1 = exp( - distance );                  % 热核函数
distance = 2 - 2*d1;

end

