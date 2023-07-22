function result = svmTest(svm, testData, kertype)   
if(strcmp(kertype,'linear'))
    result.score = svm.w*testData + svm.b;
    result.Y = sign(result.score);
elseif(strcmp(kertype,'rbf'))
    w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,testData,kertype);
    result.score = w + svm.b;
    result.Y = sign(result.score);
elseif(strcmp(kertype,'hermite'))
    w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,testData,kertype);
    result.score = w + svm.b;
    result.Y = sign(result.score);
end
end