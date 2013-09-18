function [V1 WW1 aVV bVV] = updateVWWG(Z3,K,gamma20)

    %%%%%%%%%%%%%%%% Update V %%%%%%%%%%%%%%%%%%%%%%%%
    comp_nn = zeros(K,1); %comp_nnn = zeros(K-1,1);
    for kk = 1:K
        comp_nn(kk,1) = sum(Z3 == kk); 
    end

    %for kk = 1:K-1
    %    comp_nnn(kk,1) = sum(comp_nn((kk+1):(K)));
    %end
    comp_nnn = sum(comp_nn) - cumsum(comp_nn);
    
    aVV=1+comp_nn(1:K-1,:);
    bVV = gamma20+comp_nnn(1:K-1);
    V1 = betarnd(aVV, bVV);
    
    %%%%%%%%%%%%%%%%%%%% Update WW %%%%%%%%%%%%%%%%%%%
    WW1 = zeros(K,1);
    WW1(1) = V1(1); prod = 1-V1(1);    
    for rr = 2:K-1
        WW1(rr) = prod*V1(rr);
        prod = prod*(1-V1(rr));
    end
    WW1(K) = prod;
       

    

