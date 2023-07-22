function [Acc,SVs,preY,trainTime,testTime,proList,svm,maxLabel] = svm_multiclass(trainData,trainLabel,testData,testLabel,nuclass,class,kertype,C)
testY  = []; %保存最后的标签
%     struct SVM;
nn = 0;
[mTrain,nTrain] = size(trainData);
svsList = zeros(nTrain,1); %用来记录支持向量的个数，非0即支持向量
trainTime = 0;testTime =0;
for ii = 0:nuclass-1
    tic;
    nn = nn + 1;
    iclass = class(ii+1);
    iindex = find(trainLabel==iclass);
    %one v all
    jindex = find(trainLabel~=iclass);
    itrainLabel = trainLabel(iindex) - trainLabel(iindex) + 1;%one为1
    jtrainLabel = trainLabel(jindex) - trainLabel(jindex) - 1;%All为-1
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
    %求b
    [svm,sv_label] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
    svsList(sv_label(:,:))=1;%单次支持向量
    
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
%得到精度
preY = testY;
Acc = size(find(testLabel==preY))/size(testLabel);
svm.svnum = length(find(svsList==1));
SVs = svm.svnum;
proList = softmax(testYList);
end

