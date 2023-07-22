function [d] = getv4_new(M,para)
%¼ÆËãf=exp(-xi^2/2))
para = 0;
para2 = 2;
% if para == 0
%     para = 0;
%     para2 = 2;
% end

[n,m] = size(M);
d = ones(m,1);
%[nL,mL] = find(M>0.00001);
mi = length(M);
mL = [1:mi];
for i = 1:mi
    indj = mL(i);
    dis1 = M(indj);
    %d(indj,1) = exp(-dis1^2/2); $0808
    d(indj,1) = exp(-(dis1-para)^2/(para2));
    %d(indj,1) = exp(-2.*(dis1)^2); %0422
end
end
