function [X] = SetGuaNoise(X,P)
%�����˹����
% X n*d
% T��������������
% P�Ӱٷֱ�����
[nX,mX] = size(X);
%����n������
%n = round(nX*T);
%N = round(1+(nX-1)*rand(1,n));%ѡ������������
N = 1:nX;
for i = 1:mX
    %ÿһ������
    T = X(:,i);
    minT = min(T);
    maxT = max(T);
    %��������
    t = normrnd(0,P,length(N),1); % ����N*1��0-1����̬�ֲ������
    %tt = minT + (maxT-minT)*t; %����������һ��
    %tt = (maxT-minT)*t;
    X(N,i) = X(N,i) + t;
end
end

