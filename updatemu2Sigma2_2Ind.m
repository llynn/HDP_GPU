function [mu21 sigma21,C2] = updatemu2Sigma2_2Ind(Z5,x2,mu20,sigma20,m2,PPII,phis,Sigma_mu_inv,delta2,Phi2,K)

%%%%%%%%%%% update C2 %%%%%%%%%%%%%%%%%%%%%
gnpdf('adddata',2,mu20',PPII,m2',phis);
C2 = gnpdf('pdf&sample',2); %for dataset 2
%gnpdf('pdf',2,0);
%C2 = gnpdf('sample',2,0);
%gnpdf('adddata',2,mu20',PPII,m2',phis);
%C2 = gnpdf('pdf&sample',2); %for dataset 2
%gnpdf('pdf',2,0);
%C2 = gnpdf('sample',2,0);

% randomnumber = rand(K,1);
% [C2, L1] = mvpdf(mu20', PPII, m2', phis, randomnumber);
% C2 = C2+1;
mu21 = mu20; sigma21 = sigma20;
%%%%%%%%%%% update mu2 %%%%%%%%%%%%%%%%%%%%%%%%%%
for s = 1:K
    ll = find(Z5==s);
    xjk = x2(:,ll); sumxjk = sum(xjk,2);
    Sigma_bar = (Sigma_mu_inv(:,:,C2(s)) + length(ll).*inv(sigma20(:,:,s)));
    mumu = Sigma_bar\(Sigma_mu_inv(:,:,C2(s))*m2(:,C2(s)) + sigma20(:,:,s)\sumxjk);
    mu21(:,s) = mvnrnd(mumu, inv(Sigma_bar));
end
%%%%%%%%%%%%%% update Sigma2 %%%%%%%%%%%%%%%%%%%% 
for s = 1:K
    ll = find(Z5==s); xjk = x2(:,ll);
    df = delta2 +length(ll);
    tau = Phi2+(xjk-repmat(mu21(:,s),1,length(ll)))*(xjk-repmat(mu21(:,s),1,length(ll)))';
    tau = (tau+tau')/2;
    sigma21(:,:,s) = iwishrnd(tau,df);
end   