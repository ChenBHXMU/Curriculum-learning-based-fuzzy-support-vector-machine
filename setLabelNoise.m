function [Label] = setLabelNoise(Label,P)
%给训练数据的标签加噪声，即标签是错的
%   Label 原始标签n*1，P百分比 0-1
nLabel = length(Label);;
%产生n个样本
n = round(P*nLabel);
N = round(1+(nLabel-1)*rand(1,n));%选出加噪声样本
%标签类别
classList = unique(Label);
nClass = length(classList);
for i = 1:n
    Y = Label(N(i));
    Ytest = classList(round(1+(nClass-1)*rand(1,1)));%加噪声的标签
    while Y == Ytest
         Ytest = classList(round(1+(nClass-1)*rand(1,1)));%加噪声的标签
    end
    Label(N(i)) = Ytest;
end
end

