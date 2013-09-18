clear all
clear mex

nvmex2010b('gnpdf.cu');


%%%% to Upload data Y, and partition Y into X1 and X2
%!!! n: number of data, p: number of dimension, you can change them for testing
%Y = X3D_161;
n = 1000;
p = 10;
Y =  mvnrnd(zeros(p,1),5.*eye(p),n); 

X = zeros(n,p); %n: number of observations, p: dimension
X(:,1:4) = Y(:,[5,6,7,10]);
X(:,5:10) = Y(:,setdiff(1:p, [5,6,7,10]));
mean_X = zeros(p,1); std_X = zeros(p,1);
for i= 1:p
mean_X(i) = mean(X(:,i));
std_X(i) = std(X(:,i));
X(:,i) = (X(:,i)-mean(X(:,i)))/std(X(:,i));
end
X2 = X(:,1:4);
X1 = X(:,5:10);
x = X';
x1 = X1';
x2 = X2';

%%% !!!!You can change J and K for the test: they are the number of components in biomarker and multimer spaces
%%% SP, SPU, SPV, SPV1 shouldn't be changed. 
J = 10;
K = 16;
SP = J;
SPU = J-1;
SPV=K-1;
SPV1 = K-1;
%p = 4;
initG_2Ind;
clear C2 S gg high j jj kk len ll loca_up low lp nn_pi
clear partition pi1 prod1 rr select_up s nnk

profile on
%initilize GPU library and data
gnpdf('setdevice',1,532);
gnpdf('adddata',0,X1,W1(:,1),mu1(:,:,1)',Sigma1(:,:,:,1));
gnpdf('adddata',1,X2,1/K*ones(K,1),mu2(:,:,1)',Sigma2(:,:,:,1));


for tt = 2:500 %T
    tt 
    tic
    iter2G_2Ind;
    toc            
end
gnpdf('clear');
profile viewer


