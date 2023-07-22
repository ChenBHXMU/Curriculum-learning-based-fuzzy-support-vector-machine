function [Acc,SVs,preY,trainTime,testTime] = svmTrain( trainData,trainLabel,testData,testLabel,kertype,C)
%trainData dim*n  trainLabel 1*n
%使标签为[-1,1]
SVs = 0;%支持向量个数
class = unique(trainLabel);
nuclass = length(class);
trainTime = 0;
testTime = 0;
if(nuclass == 2)
    trainLabel = mapminmax(trainLabel,-1,1);
    index1 = find(trainLabel == 1); index2 = find(trainLabel == -1);
    trainData = [trainData(:,index1),trainData(:,index2)];trainLabel = [trainLabel(:,index1),trainLabel(:,index2)];
    tic;
    options=optimset;
    options.LargerScale='off';
    options.Display='off';
    n=length(trainLabel);
    H=(trainLabel'*trainLabel).*kernel(trainData,trainData,kertype);
    f=-ones(n,1);
    A=[];
    b=[];
    Aeq=trainLabel;
    beq=0;
    lb=zeros(n,1);
    ub=C*ones(n,1);
    a0=zeros(n,1);
    
    [a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
    [svm] = calculate_rho(a,trainData',trainLabel',C,kertype);
    epsilon=1e-5;
    sv_label=find(abs(a)>epsilon);
    svm.a=a(sv_label);
    svm.Xsv=trainData(:,sv_label);
    svm.Ysv=trainLabel(sv_label);
    SVs = SVs + svm.svnum;
    trainTime = toc;
    tic;
    [result]=svmTest(svm,testData,kertype);
    testTime = toc;
    index1 = find(result.Y > 0); index2 = find(result.Y <= 0);
    maxL = max(testLabel); minL = min(testLabel);
    result.Y(index1) = maxL;result.Y(index2) = minL;
    preY = result.Y;
    
    Acc = size(find(result.Y==testLabel))/size(testLabel);  
    toc;
    %obj = sum(C.*D) + norm(w,2)*0.5;
    %Cov = sum(obj);
    
end
if(nuclass > 2)
    testYList = [];
    testY  = []; %保存最后的标签
    %     struct SVM;
    nn = 0;
    for ii = 0:nuclass-1
        for jj = ii + 1:nuclass-1
            nn = nn + 1;
            iclass = class(ii+1);
            jclass = class(jj+1);
            iindex = find(trainLabel==iclass);
            jindex = find(trainLabel==jclass);
            itrainLabel = trainLabel(iindex);
            jtrainLabel = trainLabel(jindex) - trainLabel(jindex);
            itrainData = trainData(:,iindex);
            jtrainData = trainData(:,jindex);
            ijtrainLabel = [itrainLabel,jtrainLabel];
            ijtrainLabel = mapminmax(ijtrainLabel,-1,1);
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
            [svm] = calculate_rho(a,ijtrainData',ijtrainLabel',C,kertype);
            epsilon=1e-8;
            sv_label=find(abs(a)>epsilon);
            svm.a=a(sv_label);
            svm.Xsv=ijtrainData(:,sv_label);
            svm.Ysv=ijtrainLabel(sv_label);
            SVs = SVs + svm.svnum;
            trainTime = trainTime + toc;
            tic;
            result=svmTest(svm,testData,kertype);
            testTime = testTime + toc;
            Y = result.Y;
            iY = trainLabel(iindex(1));
            jY = trainLabel(jindex(1));
            iYindex = find(Y > 0);
            jYindex = find(Y<=0);
            %if(iY>jY)  % 这里是给真实标签
            testYList(nn,iYindex) = iY;
            testYList(nn,jYindex) = jY;
        end
    end
    %确定值
    [My,Ny] = size(testYList);
    for ii = 1:Ny
        table = unique( testYList(:,ii));
        %统计元素出现次数
        hTable = histc( testYList(:,ii), table);
        %获取出现次数最多的元素的下标，idx存放出现次数最多元素的下标，若有多个元素则返回第一个元素的下标
        [maxCount, idx] = max(hTable);
        testY(ii) = table(idx);
    end
    %得到精度
    preY = testY;
    Acc = size(find(testLabel==preY))/size(testLabel);
    svm.svnum = SVs;
end

end

