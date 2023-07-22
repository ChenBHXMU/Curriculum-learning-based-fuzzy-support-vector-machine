function [X,y] = moon()
%UNTITLED6 此处显示有关此函数的摘要
%   此处显示详细说明
noise = 0.1;
N1 = 110;
N2 = 110;
level = 0.45;
upright = 0.3;

t = pi:-pi/(N1-1):0;
X(1:N1,1) = cos(t)'+randn(N1,1)*noise-level;
X(1:N1,2) = sin(t)'+randn(N1,1)*noise-upright;

t = pi:pi/(N2-1):2*pi;
X(N1+1:N1+N2,1) = cos(t)'+randn(N2,1)*noise+level;
X(N1+1:N1+N2,2) = sin(t)'+randn(N2,1)*noise+upright;


y = [ones(N1,1); -1*ones(N2,1)];

figure

plot(X(:,1),X(:,2),'.k','Markersize',18)
hold on
plot(X(y == 1,1),X(y == 1,2),'.r','Markersize',18)
hold on
plot(X(y == -1,1),X(y == -1,2),'.b','Markersize',18)
hold on
axis equal
%title(['',num2str(str)],'Interpreter','none')
box off

end

