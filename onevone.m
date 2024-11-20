function [Acc,SVs,preY,trainTime,testTime,proList,svm,maxLabel,objFuc] = onevone(trainData,trainLabel,testData,testLabel,nuclass,class,kertype,C,type,isCluster,item,para)
%支持多分类

%     struct SVM;
epsilon = 1e-6;
nn = 0;
[~,nTrain] = size(trainData);
[mTest,nTest] = size(testData);
svsList = zeros(nTrain,1); %用来记录支持向量的个数，非0即支持向量
trainTime = 0;
testTime = 0;
testYList = zeros(nuclass,nTest);

for ii = 0:nuclass-2
    for jj = (ii+1):nuclass-1
        objFuc = zeros(item,1);
        tic;
        nn = nn + 1;
        iclass = class(ii+1);
        jclass = class(jj+1);
        iindex = find(trainLabel==iclass);
        %one v one
        jindex = find(trainLabel==jclass);
        itrainLabel = trainLabel(iindex) - trainLabel(iindex) + 1;%one为1
        jtrainLabel = trainLabel(jindex) - trainLabel(jindex) - 1;%others为-1        
        pN = length(iindex); nN = length(jindex);

        if isCluster
            itrainData = trainData(:,iindex);
            jtrainData = trainData(:,jindex);

            [clustLabel1] = dbscan(itrainData',kertype);
            [clustLabel2] = dbscan(jtrainData',kertype);
            ijtrainData = [trainData(:,iindex(clustLabel1==1)),trainData(:,jindex(clustLabel2==1))];
            itrainLabel = trainLabel(iindex(clustLabel1==1)) - trainLabel(iindex(clustLabel1==1)) + 1;%one为1
            jtrainLabel = trainLabel(jindex(clustLabel2==1)) - trainLabel(jindex(clustLabel2==1)) - 1;%All为-1
            ijtrainLabel = [itrainLabel,jtrainLabel];
        else
            itrainData = trainData(:,iindex);
            jtrainData = trainData(:,jindex);
            ijtrainLabel = [itrainLabel,jtrainLabel];
            ijtrainData = [itrainData,jtrainData];
        end
        
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
        ub=C;
        a0=zeros(n,1);
        
        [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
        %求b
        [svm,sv_label] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
        
        if isCluster
            itrainLabel = trainLabel(iindex) - trainLabel(iindex) + 1;%one为1
            jtrainLabel = trainLabel(jindex) - trainLabel(jindex) - 1;%All为-1
            itrainData = trainData(:,iindex);
            jtrainData = trainData(:,jindex);
            ijtrainLabel = [itrainLabel,jtrainLabel];
            ijtrainData = [itrainData,jtrainData];
        end
        
        i = 1;
        n=length(ijtrainLabel);
        H=(ijtrainLabel'*ijtrainLabel).*kernel(ijtrainData,ijtrainData,kertype);
        f=-ones(n,1);
        A=[];
        b=[];
        Aeq=ijtrainLabel;
        beq=0;
        lb=zeros(n,1);
        a0=zeros(n,1);
        [di,~] = getSD(svm, ijtrainData, ijtrainLabel, kertype,type,pN,nN);  
        while(i<=item)
            
            ub=di.*(C*ones(n,1));
            [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
            [svm,sv_label] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
            [ei] = getEi(svm, ijtrainData, ijtrainLabel, kertype);
            %converage
            w2 = norm(svm.w,2)^2;
            softloss = di.*ei';
            %objFuc(i) = 0.5*w2 + C * sum(softloss);
            objFuc(i) = getObjFun(a,trainData,trainLabel,kertype,C,softloss);
            if(item == 1)
                break;
            end
            
            if(i>1 && abs(objFuc(i)-objFuc(i-1))<=epsilon && objFuc(i)<=objFuc(i-1))
                i=i
                objFuc((i+1):item) = objFuc(i);
                break;
            elseif(i>1 && objFuc(i)>objFuc(i-1))
                i = i -1;
                objFuc((i+1):item) = objFuc(i);
                svm = svmsvm;
                sv_labels = sv_label;
                break;
            else
                [di] = getS(ei,type,pN,nN,para);
                %stepsize = 1;
                svmsvm = svm;
                %sv_labels = sv_label;
                i = i + 1;
            end
        end
        
        svsList(sv_label(:,:))=1;%单次支持向量
        trainTime = trainTime + toc;
        tic;
        result=svmTest_multiclass(svm,testData,kertype);
        testTime = testTime + toc;
        %testYList(nn,:) = result.score;
        indPreYmin = find(result.score<0);
        testYList((jj+1),indPreYmin) =  testYList((jj+1),indPreYmin) + 1;
        indPreYmax = find(result.score>0);
        testYList((ii+1),indPreYmax) =  testYList((ii+1),indPreYmax) + 1;
    end
    [maxLabel,maxIndex] = max(testYList);
    testY = class(maxIndex);
    %得到精度
    preY = testY;
    Acc = size(find(testLabel==preY))/size(testLabel);
    %Acc = Gmean(preY,testLabel);
    svm.svnum = length(find(svsList==1));
    SVs = svm.svnum;
    proList = softmax(testYList);
    
end

