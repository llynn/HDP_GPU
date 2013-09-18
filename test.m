clear all
clear mex

%addpath I:\matlab\Utilities\distribution
%addpath ('/home/grad/ll86/Desktop/research/Multimers_CombTet/GPU version')
%addpath ('/home/grad/ll86/Desktop/research/Multimers_CombTet/GPU_Gibbs')
nvmex2010b('gnpdf.cu');

J = 100;
K = 100;
SP = J;
SPU = J-1;
SPV=K-1;
SPV1 = K-1;
%SP = 2;
%SPU = 3;
%SPV=3;
%SPV1 = 3;
%p = 4;
%initG_2Ind;
clear C2 S gg high j jj kk len ll loca_up low lp nn_pi
clear partition pi1 prod1 rr select_up s nnk

%initilize GPU library and data
gnpdf('setdevice',1,532);
gnpdf('adddata',0,X1,W1(:,1),mu1(:,:,1)',Sigma1(:,:,:,1));
gnpdf('adddata',1,X2,1/K*ones(K,1),mu2(:,:,1)',Sigma2(:,:,:,1));

timee = zeros(1,499);
for tt = 2:500
    tt
    tic
    iter2G_2Ind;
    timee(n) = toc
end



