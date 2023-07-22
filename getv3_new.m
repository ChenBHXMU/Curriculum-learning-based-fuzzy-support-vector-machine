function [d] = getv3_new(M,para)
%¼ÆËãf=2/(1+exp(belt*xi))
%belt=0.5
para = 0.5;
if para == 0
    para = 0.5;
end
[n,m] = size(M);
d = ones(m,1);
%[nL,mL] = find(M>0.00001);
mi = length(M);
mL = [1:mi];
for i = 1:mi
    indj = mL(i);
    dis1 = M(indj);
    %d(indj,1) = 2/(1+exp(0.8*dis1));
    d(indj,1) = 2/(1+exp(para*dis1));
    %d(indj,1) = 2/(1+exp(0.1*dis1));
end
end
