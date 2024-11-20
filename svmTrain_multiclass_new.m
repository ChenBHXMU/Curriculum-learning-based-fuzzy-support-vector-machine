function [Acc,SVs,preY,trainTime,testTime,lossList,svm,maxLabel,objFuc,AccTrain,AccTest] = svmTrain_multiclass_new( trainData,trainLabel,testData,testLabel,kertype,C,item,type,isCluster,isOneVone,para)
%trainData dim*n  trainLabel 1*n
%isCluster represets whether to use curriculum learning strategies
%if isempty(para)
   para = 1; 
%end

SVs = 0;%Ö§³ÖÏòÁ¿¸öÊý
class = unique(trainLabel);
nuclass = length(class);
trainTime = 0;
testTime = 0;

objFuc = zeros(item,1);
lossList = zeros(item,1);
AccTrain = zeros(item,1);
AccTest = zeros(item,1);
[mTest,nTest] = size(testData);
testYList = zeros(nuclass,nTest);
proList = zeros(nuclass,nTest);
stepsize = 0.1;
svmsvm = [];
epsilon=1e-5;

%preList = zeros(nuclass,nTrain);
if(nuclass == 2)
    preY = zeros(1,nTest);
    trainLabel = mapminmax(trainLabel,-1,1);
    index1 = find(trainLabel == 1); index2 = find(trainLabel == -1);
    pN = length(index1); nN = length(index2);

    if isCluster
        [clustLabel1] = dbscan(trainData(:,index1)',kertype);
        [clustLabel2] = dbscan(trainData(:,index2)',kertype);
        trainData1 = [trainData(:,index1),trainData(:,index2)]; trainLabel1 = [trainLabel(:,index1),trainLabel(:,index2)];
        trainData = [trainData(:,index1(clustLabel1==1)),trainData(:,index2(clustLabel2==1))];
        trainLabel = [trainLabel(:,index1(clustLabel1==1)),trainLabel(:,index2(clustLabel2==1))];
    else
        trainData = [trainData(:,index1),trainData(:,index2)];trainLabel = [trainLabel(:,index1),trainLabel(:,index2)];
    end
    
    options=optimset;
    options.LargerScale='off';
    options.Display='off';
    n=length(trainLabel);
    H=(trainLabel'*trainLabel).*kernel(trainData,trainData,kertype);
    %H = (H+H')./2;
    f=-ones(n,1);
    A=[];
    b=[];
    Aeq=trainLabel;
    beq=0;
    lb=zeros(n,1);
    ub=C*ones(n,1);
    a0=zeros(n,1);
    tic;
    [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    [svm,~] = calculate_rho(a,trainData',trainLabel',C,kertype);
    if isCluster
        trainData = trainData1; trainLabel = trainLabel1;
    end

    n=length(trainLabel);
    H=(trainLabel'*trainLabel).*kernel(trainData,trainData,kertype);
    %H = (H+H')./2;
    f=-ones(n,1);
    Aeq=trainLabel;
    beq=0;
    lb=zeros(n,1);
    a0=zeros(n,1);
    i = 1;
    [di,loss] = getSD(svm, trainData, trainLabel, kertype, type,pN,nN,para);
    while(i<=item) 
        ub=di.*(C*ones(n,1));
        [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);       
        [svm,sv_label] = calculate_rho(a,trainData',trainLabel',C,kertype);
        [ei] = getEi(svm, trainData, trainLabel, kertype);
        w2 = norm(svm.w,2)^2;
        softloss = di.*ei';
        lossList(i) = sum(ei);
        %objFuc(i) = 0.5*w2 + C * sum(softloss);
        objFuc(i) = getObjFun(a,trainData,trainLabel,kertype,C,softloss);
        R1 = svmTest(svm, trainData, kertype);  
        AccTrain(i) = size(find(trainLabel==R1.Y))/size(trainLabel);
        maxLabel =trainLabel.*R1.score;
        R2 = svmTest(svm, testData, kertype);      
        indPreYmin = find(R2.score<0);
        preY(1,indPreYmin) = min(class);
        indPreYmax = find(R2.score>0);
        preY(1,indPreYmax) = max(class);
        AccTest(i) = size(find(testLabel==preY))/size(testLabel);

        if(item == 1)
            break;
        end
        
        if(i>1 && abs(objFuc(i)-objFuc(i-1))<=epsilon && objFuc(i)<=objFuc(i-1))
            i=i
            objFuc((i+1):item) = objFuc(i);
            lossList((i+1):item) = lossList(i);
            AccTrain((i+1):item) = AccTrain(i);
            AccTest((i+1):item) = AccTest(i);
            break;
        elseif(i>1 && objFuc(i)>objFuc(i-1))
            i = i - 1;
            objFuc((i+1):item) = objFuc(i);
            lossList((i+1):item) = lossList(i);
            AccTrain((i+1):item) = AccTrain(i);
            AccTest((i+1):item) = AccTest(i);
            svm = svmsvm;
            break;
            
        else
            svmsvm = svm;
            [di] = getS(ei,type,pN,nN,para);
            i = i + 1;
        end
    end
    SVs = SVs + svm.svnum; 
    trainTime = toc;
    tic;
    result=svmTest_multiclass(svm,testData,kertype);
    testTime = toc;
    maxLabel =testLabel.*result.score;
    indPreYmin = find(result.score<0);
    preY(1,indPreYmin) = min(class);
    indPreYmax = find(result.score>0);
    preY(1,indPreYmax) = max(class);

    
    %µÃµ½¾«¶È
    Acc = size(find(preY==testLabel))/size(testLabel);
    %Acc = Gmean(preY,testLabel);
    result = svmTest(svm, testData, kertype);  
    %get probability
    for ipro = 1:length(testLabel)
        if(preY(ipro) == min(class))
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
        [Acc,SVs,preY,trainTime,testTime,proList,svm,maxLabel,objFuc] = onevone(trainData,trainLabel,testData,testLabel,nuclass,class,kertype,C,type,isCluster,item,para);
    %else
        %[Acc,SVs,preY,trainTime,testTime,proList,svm,maxLabel,objFuc] = multiclass(trainData,trainLabel,testData,testLabel,nuclass,class,kertype,C,type,isCluster,item);
    end
end

end

