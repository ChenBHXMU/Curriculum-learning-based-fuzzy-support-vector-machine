function [Acc,SVs,preY,trainTime,testTime,proList,svm,maxLabel] = svm_multiclass(trainData,trainLabel,testData,testLabel,nuclass,class,kertype,C)
testY  = []; %�������ı�ǩ
%     struct SVM;
nn = 0;
[mTrain,nTrain] = size(trainData);
svsList = zeros(nTrain,1); %������¼֧�������ĸ�������0��֧������
trainTime = 0;testTime =0;
for ii = 0:nuclass-1
    tic;
    nn = nn + 1;
    iclass = class(ii+1);
    iindex = find(trainLabel==iclass);
    %one v all
    jindex = find(trainLabel~=iclass);
    itrainLabel = trainLabel(iindex) - trainLabel(iindex) + 1;%oneΪ1
    jtrainLabel = trainLabel(jindex) - trainLabel(jindex) - 1;%AllΪ-1
    itrainData = trainData(:,iindex);
    jtrainData = trainData(:,jindex);
    ijtrainLabel = [itrainLabel,jtrainLabel];
    ijtrainData = [itrainData,jtrainData];
    options=optimset;
    options.LargerScale='off';
    options.Display='off';
    
    n=length(ijtrainLabel);
    H=(ijtrainLabel'*ijtrainLabel).*kernel(ijtrainData,ijtrainData,kertype);
    f=-ones(n,1);
    A=[];
    b=[];
    Aeq=ijtrainLabel;
    beq=0;
    lb=zeros(n,1);
    ub=C*ones(n,1);
    a0=zeros(n,1);
    tic;
    [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    %��b
    [svm,sv_label] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
    svsList(sv_label(:,:))=1;%����֧������
    
    svm.a=a(sv_label);
    svm.Xsv=ijtrainData(:,sv_label);
    svm.Ysv=ijtrainLabel(sv_label);

    trainTime = trainTime + toc;
    tic;
    result=svmTest_multiclass(svm,testData,kertype);
    testTime = testTime + toc;
    testYList(nn,:) = result.score;

end
[maxLabel,maxIndex] = max(testYList);
testY = class(maxIndex);
%�õ�����
preY = testY;
Acc = size(find(testLabel==preY))/size(testLabel);
svm.svnum = length(find(svsList==1));
SVs = svm.svnum;
proList = softmax(testYList);
end

