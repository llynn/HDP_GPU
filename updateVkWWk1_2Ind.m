function [Vk1 WWk1 accept_V1] = updateVkWWk1_2Ind(Z5,Z2,WW,alpha2,J,K,Vk0,mu11,sigma11,W11, mu20,sigma20,WWk0)
Vk1 = Vk0; Vk1_prop = Vk0;
WWk1 = WWk0; WWk1_prop = WWk0; 
WWsum = cumsum(WW);
nnk = zeros(K,1); accept_V1 = zeros(J,1);
% ZZ5 =  full(sparse((1:n)',double(Z5),1,n,K));
for j = 1:J    
    lj = find(Z2==j);
    z2j = Z5(lj);
    for kk = 1:K
        nnk(kk) = sum(z2j==kk); 
    end
    nnksum = sum(nnk)-cumsum(nnk);
    betamatrix1 = alpha2.*WW(1:K-1)+nnk(1:K-1);
    betamatrix2 = alpha2.*(1-WWsum(1:K-1))+nnksum(1:K-1);
    Vk1_prop(j,:)= double(betarnd(betamatrix1,betamatrix2));
    
    prod1 = (1-Vk1_prop(j,1)); WWk1_prop(j,1) = Vk1_prop(j,1);
    for kk = 2:K-1
        WWk1_prop(j,kk) = prod1*Vk1_prop(j,kk);
        prod1 = prod1*(1-Vk1_prop(j,kk));
    end
    WWk1_prop(j,K) = prod1;
    QQ = sum(betapdf_log(Vk1_prop(j,:)', alpha2*WW(1:K-1), alpha2*(1-cumsum(WW(1:K-1)))))...
        - sum(betapdf_log(Vk0(j,:)', alpha2*WW(1:K-1), alpha2*(1-cumsum(WW(1:K-1)))));
    QQ = QQ + sum(betapdf_log(Vk0(j,:)', betamatrix1, betamatrix2))- sum(betapdf_log(Vk1_prop(j,:)', betamatrix1, betamatrix2));
    gnpdf('updatecluster',0,W11,mu11',sigma11);
        gnpdf('updatecluster',1,1/K*ones(K,1),mu20',sigma20);
         WWk1(WWk1<0.00001) = 0.00001;
        WWk1 = WWk1./repmat(sum(WWk1,2),1,K);
        %denP = gnpdf('z5',0,1,WWk1,1);
        %denP = sum(denP.*ZZ5,2);
         gnpdf('z5',0,1,WWk1,1);
        sum_denP = gnpdf('sum',1,double(Z5));
        gnpdf('updatecluster',0,W11,mu11',sigma11);
        gnpdf('updatecluster',1,1/K*ones(K,1),mu20',sigma20);
           WWk1_prop(WWk1_prop<0.00001) = 0.00001;
        WWk1_prop = WWk1_prop./repmat(sum(WWk1_prop,2),1,K);
        %denP_prop = gnpdf('z5',0,1,WWk1_prop,1);
        %denP_prop = sum(denP_prop.*ZZ5,2);
        gnpdf('z5',0,1,WWk1_prop,1);
        sum_denP_prop = gnpdf('sum',1,double(Z5));
        QQ = QQ+sum_denP_prop-sum_denP;
         if log(rand(1))<QQ
             accept_V1(j) = 1;
            WWk1(j,:) = WWk1_prop(j,:);
            Vk1(j,:) = Vk1_prop(j,:);
         else
            Vk1_prop(j,:) = Vk0(j,:);
            WWk1_prop(j,:)= WWk0(j,:);
        end
             
end
end
