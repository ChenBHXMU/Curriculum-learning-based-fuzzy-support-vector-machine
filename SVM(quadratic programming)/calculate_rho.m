function [svm,sv_label] = calculate_rho(alphas,trainData,trainLabel,C,kertype)
%计算偏移b trainData n*dim trainLabel n*1
% 
epsilon=1e-6;
%% 计算权值W
sv_label=find(abs(alphas)>epsilon);
svm.svnum = length(sv_label);
%W = (alphas.*trainLabel)'*trainData;

svm.a=alphas(sv_label,:);
svm.Xsv=trainData(sv_label,:)';
svm.Ysv=trainLabel(sv_label,:)';
svm.w = (svm.a.*svm.Ysv')'*svm.Xsv';
%W = (alphas(sv_label).*trainLabel(sv_label))'*trainData(sv_label,:);
%计算b
ub = 10000;
lb = -10000;
svb_label1=find((abs(alphas)>epsilon & (abs(alphas)<0.999999*C)));%用二次规划求解时，C有时候是double型，近似的C
if ~isempty(svb_label1)
    temp = (alphas.*trainLabel)'*kernel(trainData',trainData(svb_label1,:)',kertype);
    b = mean(trainLabel(svb_label1)-temp');  %b取均值
else
    for i = 1 : length(alphas)
        temp = (alphas.*trainLabel)'*kernel(trainData',trainData(i,:)',kertype);
        if alphas(i) >= 0.9999*C
            if trainLabel(i) == -1
                ub = min(ub,temp'-trainLabel(i));
            else
                lb = max(lb,temp'-trainLabel(i));
            end
        end
        if alphas(i) <= epsilon
            if trainLabel(i) == 1
                ub = min(ub,temp'-trainLabel(i));
            else
                lb = max(lb,temp'-trainLabel(i));
            end
        end
    end
    b = -(ub+lb)/2;  %b取均值
end

svm.b = b;

end

