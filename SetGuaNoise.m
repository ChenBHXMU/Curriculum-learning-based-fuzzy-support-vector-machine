function [X] = SetGuaNoise(X,P)
%制造高斯噪声
% X n*d
% T多少样本有噪声
% P加百分比噪声
[nX,mX] = size(X);
%产生n个样本
%n = round(nX*T);
%N = round(1+(nX-1)*rand(1,n));%选出加噪声样本
N = 1:nX;
for i = 1:mX
    %每一列特征
    T = X(:,i);
    minT = min(T);
    maxT = max(T);
    %加入噪声
    t = normrnd(0,P,length(N),1); % 产生N*1，0-1的正态分布随机数
    %tt = minT + (maxT-minT)*t; %跟样本量纲一致
    %tt = (maxT-minT)*t;
    X(N,i) = X(N,i) + t;
end
end

