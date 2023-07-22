function [Acc,SVs,TrainTime,TestTime,proList,svm,margin,objFuc,AccTrain,AccTest] = testmySVM_new(trainData,trainLabel,testData,testLabel,kertype,C,item,type,isCluster,isOneVone)
%trainData=num*dim ¡£trainLabel = num*dim 

[Acc,SVs,preY,TrainTime,TestTime,proList,svm,margin,objFuc,AccTrain,AccTest] = svmTrain_multiclass_new( trainData',trainLabel',testData',testLabel',kertype,C,item,type,isCluster,isOneVone);

end

