function [Vk1 WWk1] = updateVkWWk1_2IndG(Z5,Z2,WW,alpha2,J,K)
Vk1 = zeros(J,K-1);
WWk1 = zeros(J,K);
WWsum = cumsum(WW);
nnk = zeros(K,1);
for j = 1:J    
    lj = find(Z2==j);
    z2j = Z5(lj);
    for kk = 1:K
        nnk(kk) = sum(z2j==kk); 
    end
    nnksum = sum(nnk)-cumsum(nnk);
    betamatrix1 = alpha2.*WW(1:K-1)+nnk(1:K-1);
    betamatrix2 = alpha2.*(1-WWsum(1:K-1))+nnksum(1:K-1);
    Vk1(j,:)= double(betarnd(betamatrix1,betamatrix2));
    
    prod1 = (1-Vk1(j,1)); WWk1(j,1) = Vk1(j,1);
    for kk = 2:K-1
        WWk1(j,kk) = prod1*Vk1(j,kk);
        prod1 = prod1*(1-Vk1(j,kk));
    end
    WWk1(j,K) = prod1;
end
end
