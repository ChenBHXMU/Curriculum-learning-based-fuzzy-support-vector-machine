function [distance] = getHermiteDis(X)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%X= X';
[~,N] = size(X);
distance = zeros(N,N);
% X2 = sum( X.^2 , 1 );
% distance = repmat( X2 , N , 1 ) + repmat( X2' , 1 , N ) - 2 * X' * X;  % 2�����������
% d1 = exp( - distance );                  % �Ⱥ˺���
% distance = 2 - 2*d1;

for i = 1:N
    for j = 1:N
        K1 = kernel(X(:,i),X(:,i),'hermite');
        K3 = kernel(X(:,j),X(:,j),'hermite');
        K2 = kernel(X(:,i),X(:,j),'hermite');
        distance(i,j) = K1 - 2.*K2 + K3;
    end
end

% XXX = X.^3-3.*X; 
% XX = X.^2 -1; 
% d1 = XXX'*XXX + XX'*XX + X'*X + 1;
% d2 = kernel(X,X,'hermite');
% distance = 2.*d1 - 2.*d2;
end

