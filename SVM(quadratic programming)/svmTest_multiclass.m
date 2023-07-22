function result = svmTest_multiclass(svm, testData, kertype)   
if(strcmp(kertype,'linear'))
    result.score = svm.w*testData + svm.b;
else
    w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,testData,kertype);
    result.score = w + svm.b;
end
end