Z5 = SampleZ5(mu1(:,:,tt-1)',Sigma1(:,:,:,tt-1),mu2(:,:,tt-1)',Sigma2(:,:,:,tt-1),W1(:,tt-1),WWk(:,:,tt-1));

[Z1 Z2 ] = updateZ1Z2_2IndG(W1(:,tt-1),mu1(:,:,tt-1),Sigma1(:,:,:,tt-1),...
     WWk(:,:,tt-1),mu2(:,:,tt-1),Sigma2(:,:,:,tt-1),Z5);
 


%[U1(:,tt) W1(:,tt)] = updateUW_newG(Z1,Z4,J,alpha1(tt-1));
[U1(:,tt) W1(:,tt)] = updateUW_2IndG(Z1,Z2,J,alpha1(tt-1),W1(:,tt-1));


alpha1(tt) = gamrnd(J+e1-1, 1/(f1-sum(log(1-U1(:,tt)))));

%[mu1(:,:,tt) Sigma1(:,:,:,tt)]=updatemu1Sigma1_newG(J,Z1,Z4,X1,...
 %   lambda1,m1,delta1,Phi1, Sigma1(:,:,:,tt-1),p1);

[mu1(:,:,tt) Sigma1(:,:,:,tt)]=updatemu1Sigma1_2IndG(J,Z1,Z2,X1,lambda1,m1,delta1,Phi1,...
    mu1(:,:,tt-1),Sigma1(:,:,:,tt-1));


[V(:,tt) WW(:,tt) aVV bVV] = updateVWW_2IndG(K,gamma2(tt-1), WW(:,tt-1),mu2(:,:,tt-1),Sigma2(:,:,:,tt-1));  
%[V(:,tt) WW(:,tt) accept_V(:,tt)] = updateVWWG_MH(K,J,gamma2(tt-1),Vk(:,:,tt-1),alpha2, V(:,tt-1),WW(:,tt-1), adp_var);

gamma2(tt) = gamrnd(K+ee-1, 1/(ff-sum(log(1-V(:,tt)))));

[Vk(:,:,tt) WWk(:,:,tt)] = updateVkWWk1_2IndG(Z5,Z2,WW(:,tt),alpha2,J,K);

[mu2(:,:,tt) Sigma2(:,:,:,tt),C2] = updatemu2Sigma2_2IndG(Z5,x2,mu2(:,:,tt-1),Sigma2(:,:,:,tt-1),m2,PPII,phis,Sigma_mu_inv,delta2,Phi2,K);