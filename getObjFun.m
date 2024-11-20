function value = getObjFun(a,trainData,trainLabel,kertype,C,loss)
% a = a ./ max(a);
k = kernel(trainData,trainData,kertype);
aa = a*a';
yy = trainLabel'*trainLabel;

w = aa.*yy.*k;

% w2 = norm(w,2)^2;
% w2 = norm(w,"fro")^2;
w2 = sum(w(:));
value = 0.5*w2 + C*sum(loss);
end