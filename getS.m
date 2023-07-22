function [s] = getS(M,f,pN,nN,para)

switch f
    case {1}
        s = getv1_new(M);
    case {2}
        s = getv2_new(M,pN,nN);
    case {3}
        s = getv3_new(M,para);
    case {4}
       s = getv4_new(M,para);

end

end

