function [d] = getv1_new(M)
%¼ÆËãf=1/(1+|xi|)
%belt=0.5
[n,m] = size(M);
d = ones(m,1);
%[nL,mL] = find(M>0.00001);
mi = length(M);
mL = [1:mi];
for i = 1:mi
    indj = mL(i);
    dis1 = M(indj);
    d(indj,1) = 1/(1+abs(dis1));
end
end
