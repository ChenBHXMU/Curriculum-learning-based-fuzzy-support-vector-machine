function [Acc,SVs,preY,trainTime,testTime,proList,svm,maxLabel] = svmTrain_multiclss( trainData,trainLabel,testData,testLabel,kertype,C,isOneVone)
%trainData dim*n  trainLabel 1*n
%分类y=max(f(x))
%使标签为[-1,1]
SVs = 0;%支持向量个数
class = unique(trainLabel);
nuclass = length(class);
trainTime = 0;
testTime = 0;
[mTest,nTest] = size(testData);
testYList = zeros(nuclass,nTest);
proList = zeros(nuclass,nTest);
epsilon=1e-5;
if(nuclass == 2)
    preY = zeros(1,nTest);
    trainLabel = mapminmax(trainLabel,-1,1);
    index1 = find(trainLabel == 1); index2 = find(trainLabel == -1);
    trainData = [trainData(:,index1),trainData(:,index2)];trainLabel = [trainLabel(:,index1),trainLabel(:,index2)];
    options=optimset;
    options.LargerScale='off';
    options.Display='off';
    n=length(trainLabel);
    tic;
    H=(trainLabel'*trainLabel).*kernel(trainData,trainData,kertype);
    H = (H+H')./2;
    f=-ones(n,1);
    A=[];
    b=[];
    Aeq=trainLabel;
    beq=0;
    lb=zeros(n,1);
    ub=C*ones(n,1);
    a0=zeros(n,1);
    
    [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    [svm,sv_label] = calculate_rho(a,trainData',trainLabel',C,kertype);

    SVs = SVs + svm.svnum;
    trainTime = toc;
%     tic;

    tic;
    result=svmTest_multiclass(svm,testData,kertype);
    testTime = toc;
    maxLabel =testLabel.*result.score;
    indPreYmin = find(result.score<0);
    preY(1,indPreYmin) = min(class);
    indPreYmax = find(result.score>0);
    preY(1,indPreYmax) = max(class);
    %得到精度
    Acc = size(find(preY==testLabel))/size(testLabel);
    %Acc = Gmean(preY,testLabel);
    result = svmTest(svm, testData, kertype);  
    %get probability
    for ipro = 1:length(testLabel)
        if(preY(ipro) == -1)
            proList(1,ipro) = getProbability(result.score(ipro));
            proList(2,ipro) = 1- proList(1,ipro);
        else
            proList(2,ipro) = getProbability(result.score(ipro));
            proList(1,ipro) = 1- proList(2,ipro);
        end    
    end
end
if(nuclass > 2)
    if isOneVone
        [Acc,SVs,preY,trainTime,testTime,proList,svm,maxLabel] = svm_onevone(trainData,trainLabel,testData,testLabel,nuclass,class,kertype,C);
    else
        [Acc,SVs,preY,trainTime,testTime,proList,svm,maxLabel] = svm_multiclass(trainData,trainLabel,testData,testLabel,nuclass,class,kertype,C);
    end
end

end

