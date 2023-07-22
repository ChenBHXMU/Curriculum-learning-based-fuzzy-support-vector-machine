function [AccALL] = getAllSVM(trainData,trainLabel,testData,testLabel,C,kertype)
[N,M] =size(trainData);
isOneVone = 1;
isCluster = 1;
item = 20;
type = 1;
[Acc1,SVs1] = testmySVM_new(trainData,trainLabel,testData,testLabel,kertype,C,item,type,isCluster,isOneVone);
type = 2;
[Acc2,SVs2] = testmySVM_new(trainData,trainLabel,testData,testLabel,kertype,C,item,type,isCluster,isOneVone);

type = 3;
[Acc3,SVs3] = testmySVM_new(trainData,trainLabel,testData,testLabel,kertype,C,item,type,isCluster,isOneVone);
type = 4;
[Acc4,SVs4] = testmySVM_new(trainData,trainLabel,testData,testLabel,kertype,C,item,type,isCluster,isOneVone);

AccALL = [Acc1,Acc2,Acc3,Acc4];


end

