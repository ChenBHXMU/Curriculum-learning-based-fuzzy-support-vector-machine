function [ei] = getEi(svm, Xt, Yt, kertype)
%���di

w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,Xt,kertype); 

result.score = w + svm.b;  
ei = max(0,1-Yt.*result.score);


