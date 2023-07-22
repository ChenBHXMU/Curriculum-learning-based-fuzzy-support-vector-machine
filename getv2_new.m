function [d] = getv2_new(M,pN,nN)
%¼ÆËãf=1-1/max(xi)
[n,m] = size(M);
d = ones(m,1);
%[nL,mL] = find(M>0.00001);
mi = length(M);
mL = [1:mi];
epsilon=0.00001;
maxpd = max(M(1:pN))+epsilon;
maxnd = max(M(pN+1:m))+epsilon;
%maxd = max(M);
for i = 1:mi
    indj = mL(i);
        if indj <= pN
            dis1 = M(indj);
            d(indj,1) = 1 - dis1/maxpd;
        else
            dis1 = M(indj);
            d(indj,1) = 1 - dis1/maxnd;
        end
%     dis1 = M(indj);
%     d(indj,1) = 1 - dis1/maxd;
end
end

