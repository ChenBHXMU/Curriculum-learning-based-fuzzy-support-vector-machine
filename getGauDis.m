function [distance] = getGauDis(X)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%X= X';
[~,N] = size(X);
distance = zeros(N,N);
X2 = sum( X.^2 , 1 );
distance = repmat( X2 , N , 1 ) + repmat( X2' , 1 , N ) - 2 * X' * X;  % 2�����������
d1 = exp( - distance );                  % �Ⱥ˺���
distance = 2 - 2*d1;

end

