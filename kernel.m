function K = kernel( X,Y,type )
%X dim*n
switch type
    case 'linear'
        K=X'*Y;
    case 'rbf'
        delta=1;
        delta=delta*delta;
        XX=sum(X'.*X',2);
        YY=sum(Y'.*Y',2);
        XY=X'*Y;
        K=abs(repmat(XX,[1 size(YY,1)])+repmat(YY',[size(XX,1) 1])-2*XY);
        K=exp(-K./delta);
    case 'hermite'
        %n = 3
        XXX = X.^3-3.*X; YYY = Y.^3-3.*Y;
        XX = X.^2 -1; YY = Y.^2 - 1;
        K = XXX'*YYY + XX'*YY + X'*Y + 1;
end
end