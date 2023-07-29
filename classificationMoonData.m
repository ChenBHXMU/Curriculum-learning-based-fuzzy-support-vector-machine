function classificationMoonData()
[X,y] = moon();

kertype = 'rbf'; %rbf or linear or hermite
C = 100;
isOneVone = 1;
[Acc1,SVs,preY,trainTime,testTime,proList,svm1,maxLabel]  = svmTrain_multiclss(X',y',X',y',kertype,C,isOneVone);
figure
plotMoon(X,y,svm1,kertype)

[yy] = setLabelNoise(y,0.2);
XX = SetGuaNoise(X,0.5);
figure
[Acc2,SVs,preY,trainTime,testTime,proList,svm2,maxLabel]  = svmTrain_multiclss(XX',yy',X',y',kertype,C,isOneVone);
plotMoon(XX,yy,svm1,kertype);
getY1(svm2,kertype);
item = 20;
isCluster = 1;
type = 1;
figure
[Acc3,SVs,TrainTime,TestTime,lossList,svm3,margin,objFuc,AccTrain,AccTest] = testmySVM_new(X,yy,X,y,kertype,C,item,type,isCluster,isOneVone);
plotMoon(XX,yy,svm1,kertype);
getY1(svm3,kertype);
type = 2;
figure
[Acc4,SVs,TrainTime,TestTime,lossList,svm4,margin,objFuc,AccTrain,AccTest] = testmySVM_new(X,yy,X,y,kertype,C,item,type,isCluster,isOneVone);
plotMoon(XX,yy,svm1,kertype);
getY1(svm4,kertype);
type = 3;
figure
[Acc5,SVs,TrainTime,TestTime,lossList,svm5,margin,objFuc,AccTrain,AccTest] = testmySVM_new(X,yy,X,y,kertype,C,item,type,isCluster,isOneVone);
plotMoon(XX,yy,svm1,kertype);
getY1(svm5,kertype);
type = 4;
figure
[Acc6,SVs,TrainTime,TestTime,lossList,svm6,margin,objFuc,AccTrain,AccTest] = testmySVM_new(X,yy,X,y,kertype,C,item,type,isCluster,isOneVone);
plotMoon(XX,yy,svm1,kertype);
getY1(svm6,kertype);


% 
[Acc1,Acc2,Acc3,Acc4,Acc5,Acc6]

end

function plotMoon(X,y,svm,kertype)
plot(X(:,1),X(:,2),'.k','Markersize',18)
hold on
plot(X(y == 1,1),X(y == 1,2),'.r','Markersize',18)
hold on
plot(X(y == -1,1),X(y == -1,2),'.b','Markersize',18)
hold on
axis equal
box on
grid on
xlabel('x(1)');
ylabel('x(2)');
set(gca,'FontName','Times New Roman','FontSize',20,'LineWidth',1);

hold on
getY(svm,kertype);
end

function getY(SVM,kertype)
x1 = linspace(-1.5,1.5,900);
x2 = x1;
Y = zeros(length(x1),length(x2));
for i = 1:length(x1)
    for j = 1:length(x2)
        x = x1(i);
        y = x2(j);
        X = [x,y];
        result = svmTest(SVM, X', kertype);
        Y(i,j) = result.score;
        if abs(Y(i,j)) < 0.018
            plot(x,y,'.k','Markersize',5)
        end
    end
end
set(gca,'xlim',[-1.5 1.5]);
set(gca,'ylim',[-1.5 1.5]);
end

function getY1(SVM,kertype)
x1 = linspace(-1.5,1.5,850);
x2 = x1;
Y = zeros(length(x1),length(x2));
for i = 1:length(x1)
    for j = 1:length(x2)
        x = x1(i);
        y = x2(j);
        X = [x,y];
        result = svmTest(SVM, X', kertype);
        Y(i,j) = result.score;
        if abs(Y(i,j)) < 0.01
            plot(x,y,'.m','Markersize',5)
        end
    end
end
set(gca,'xlim',[-1.5 1.5]);
set(gca,'ylim',[-1.5 1.5]);
end
