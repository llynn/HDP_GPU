function [U1_prop W1_prop ] = updateUW_2IndG(Z1,Z2,J,alpha0,W10)  
    comp_nn = zeros(J,1); 
    for kk = 1:J
        comp_nn(kk,1) = sum(Z1==kk)+sum(Z2==kk);
    end
    comp_nnn = sum(comp_nn)-cumsum(comp_nn);
 
    U1_prop = betarnd(1+comp_nn(1:J-1,:), alpha0+comp_nnn(1:J-1,:));
    W1_prop = W10;
    W1_prop= U1_prop(1);  prod = (1-U1_prop(1)); 
    for kk = 2:J-1
        W1_prop(kk) = prod*U1_prop(kk);
        prod = prod*(1-U1_prop(kk));
    end
    W1_prop(J) = prod;
    

end

