function [s,D] = getSD(svm, Xt, Yt, kertype,f,pN,nN,para)
%obtain the fuzzy membership degree
w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,Xt,kertype); 

result.score = w + svm.b;  
Yts = Yt.*result.score;
D = max(0,1-Yts);%是我们的
[s] = getS(D,f,pN,nN,0);
%if(f>4)
    %[s] = getS(Yts,f,pN,nN,0);
%else
    %[s] = getS(D,f,pN,nN,0);
%end






end

