function [Label] = setLabelNoise(Label,P)
%��ѵ�����ݵı�ǩ������������ǩ�Ǵ��
%   Label ԭʼ��ǩn*1��P�ٷֱ� 0-1
nLabel = length(Label);;
%����n������
n = round(P*nLabel);
N = round(1+(nLabel-1)*rand(1,n));%ѡ������������
%��ǩ���
classList = unique(Label);
nClass = length(classList);
for i = 1:n
    Y = Label(N(i));
    Ytest = classList(round(1+(nClass-1)*rand(1,1)));%�������ı�ǩ
    while Y == Ytest
         Ytest = classList(round(1+(nClass-1)*rand(1,1)));%�������ı�ǩ
    end
    Label(N(i)) = Ytest;
end
end

