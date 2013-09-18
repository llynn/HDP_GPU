%load X1;
%load X2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T = 50000;                 %number of iterations
p1 = size(X1,2);  
p2 = p-p1; % total number of multimers
%nc = 2;                  %nc: number of color combinations;
%ncomb = nchoosek(p2,nc);
%H = nchoosek(1:p2, nc);    %possible groupings with p2 number of colors and nc color combinations 
%S = ncomb+1;           % cluster numbers

SP = J;
SPU = J-1;
SPV=K-1;
SPV1 = K-1;
accept_U1 = zeros(J,T);
accept_muSig1 = zeros(J,T);
accept_V = zeros(K,T);
accept_V1 = zeros(J,T);
%%%%%%%%%%%%% Hyperparameters %%%%%%%%%%%%%%%%%%%%%%%%
Phi1 = 10.*eye(p1); % prior for Sigma1
Phi2 = 10.*eye(p2); % prior for Sigma2

m1 = zeros(p1,1);    % hyperparameter for mean vectore of  prior for mu1
 

high = 3; low = -2;
%high1 = 5.5, high 2 = 5, high3 = 8, high4 = 6;
%low1 = -3, low2 = -3,low3 = -2, low4 = -3;
partition = [0,high,low];
lp = length(partition);
S = lp^p2; 
%PPII = (1-0.0005)/(S-1)*ones(1,S-1); % mixture proportion for the mixture prior for mu2
%PPII = [0.0005,PPII];
PPII = 1/S*ones(1,S);
m2 = npermutek(partition, p2)'; %
%m2(:,end) = high;
phis = zeros(p2,p2,S); %mixture hyper-prior on mu2  

for jj = 2:S
    phis(:,:,jj) = 10*eye(p2);
    select_up = nchoosek(m2(1:p2,jj)',2);
    loca_up = nchoosek(1:p2,2);
    len = size(select_up,1);
    for gg = 1:len
        if select_up(gg,1) == select_up(gg,2)
            phis(loca_up(gg,1),loca_up(gg,2),jj) = 5*0.6;
            phis(loca_up(gg,2),loca_up(gg,1),jj) = 5*0.6;     
        else
            phis(loca_up(gg,1),loca_up(gg,2),jj) = 5*(-0.6);
            phis(loca_up(gg,2),loca_up(gg,1),jj) = 5*(-0.6);
        end
    end                 
end
phis(:,:,1) = 10*eye(p2,p2);

e1 = 50; f1 = 1; ee = 50; ff = 1;
gamma2 = 50*ones(T,1);
delta1 = p1+1+20;
delta2 = p2+1+20;
alpha2 = 50;
lambda1 = 5;

%m2(1,(find(m2(1,:)==high))) = 6; m2(1,(find(m2(1,:)==low))) = -3;
%m2(2,(find(m2(2,:)==high))) = 5.5; m2(2,(find(m2(2,:)==low))) = -3;
%m2(3,(find(m2(3,:)==high))) = 8.5; m2(3,(find(m2(3,:)==low))) = -3;
%m2(4,(find(m2(4,:)==high))) = 6.5; m2(4,(find(m2(4,:)==low))) = -3;

%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%
alpha1 = ones(1,T);        % different alpha for each cluster
alpha1(1) = gamrnd(e1,1/(f1));


W1 = zeros(J,T); U1 = zeros(J-1,T);
U1(:,1) = betarnd(ones(J-1,1),alpha1(1),[J-1,1]);
W1(1,1) = U1(1,1); prod1 = (1-U1(1,1)); 
for kk = 2:J-1
    W1(kk,1) = prod1*U1(kk,1);
    prod1= prod1*(1-U1(kk,1));
end
W1(J,1) = prod1; 

mu1 = zeros(p1,J,T);
Sigma1 = zeros(p1,p1,J,T);
for rr = 1:J
    Sigma1(:,:,rr,1) = iwishrnd(Phi1, delta1);
    mu1(:,rr,1) = mvnrnd(m1, lambda1.*Sigma1(:,:,rr,1));
end

WW = zeros(K,T); V = zeros(K-1,T);
WWk = zeros(J,K,T); Vk = zeros(J,K-1,T);

V(:,1) = betarnd(1,gamma2(1,1),[K-1,1]);

WW(1,1) = V(1,1);
prod1 = (1-V(1,1));
for kk = 2:K-1
    WW(kk,1) = prod1*V(kk,1);
    prod1 = prod1*(1-V(kk,1));
end
WW(K,1) = prod1;
    
nn_pi = zeros(K-1,1);
for ll = 1:K-1
    nn_pi(ll) = 1-sum(WW(1:ll,1));
end
for j = 1:J
    Vk(j,:,1) = betarnd(alpha2.*WW(1:end-1,1), alpha2.*(nn_pi));
    %any = find(Vk(j,:,1) ==0);
    %Vk(j,any,1) = 0.0001;
    prod1 = (1-Vk(j,1,1)); WWk(j,1,1) = Vk(j,1,1);
    for kk = 2:K-1
        WWk(j,kk,1) = prod1*Vk(j,kk,1);
        prod1 = prod1*(1-Vk(j,kk,1));
    end
    WWk(j,K,1) = prod1;
end

mu2 = zeros(p2,K,T);
Sigma2 = zeros(p2,p2,K,T);     % Randomly generate Sigma and mu from prior
C2 = zeros(K,1);
for s = 1:K
    C2(s) = discreternd(PPII,1); % store component membership for mu2
    Sigma2(:,:,s,1) = iwishrnd(Phi2, delta2);
    mu2(:,s,1) = mvnrnd(m2(:,C2(s)),phis(:,:,C2(s)));
end

clear mex

const = -0.5*log(2*pi)*p1;
Sigma_mu_inv = zeros(p2,p2,S);
for jj = 1:S
    
    Sigma_mu_inv(:,:,jj) = inv(phis(:,:,jj));
end

nnk = zeros(K,1); 

n= size(X1,1);
x1 = X1'; x2 = X2';
