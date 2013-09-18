function [U1 W1 accept_U1] = updateUW_2Ind(Z1,Z2,Z5,J,K,alpha0,U10,W10, mu10, sigma10,SPU,mu20,sigma20,wwk) 
    comp_nn = zeros(J,1); comp_nn1 = zeros(J,1); accept_U1 = zeros(J,1);
    for kk = 1:J
        comp_nn1(kk,1) = sum(Z1==kk);
        comp_nn(kk,1) = comp_nn1(kk,1)+sum(Z2==kk);
    end
    comp_nnn = sum(comp_nn)-cumsum(comp_nn);
    comp_nnn1 = sum(comp_nn1)-cumsum(comp_nn1);
    U1_prop = U10; W1_prop = W10;
    W1 = W10; U1 = U10;
    %ZZ5 =  full(sparse((1:n)',double(Z5),1, n,K));
    wwk(wwk<0.00001) = 0.00001;
    wwk = wwk./repmat(sum(wwk,2),1,K);
    for jj = 1:(J-1)/SPU 
        rr = 1+(jj-1)*SPU:SPU+(jj-1)*SPU;
        U1_prop(rr) = betarnd(1+comp_nn(rr,:), alpha0+comp_nnn(rr,:));
        W1_prop(1)= U1_prop(1);  prod = (1-U1_prop(1)); 
        for kk = 2:J-1
            W1_prop(kk) = prod*U1_prop(kk);
            prod = prod*(1-U1_prop(kk));
        end
        W1_prop(J) = prod;
    
        QQ= sum(betapdf_log(U1_prop(rr),1+comp_nn1(rr,:), alpha0+comp_nnn1(rr,:))-...
          betapdf_log(U10(rr),1+comp_nn1(rr,:), alpha0+comp_nnn1(rr,:)));
        QQ = QQ + sum(betapdf_log(U10(rr),1+comp_nn(rr,:), alpha0+comp_nnn(rr,:))-...
          betapdf_log(U1_prop(rr),1+comp_nn(rr,:), alpha0+comp_nnn(rr,:)));
         
        gnpdf('updatecluster',0,W1,mu10',sigma10);
        gnpdf('updatecluster',1,1/K*ones(K,1),mu20',sigma20);
        gnpdf('z5',0,1,wwk,1);
        %denP = sum(sum(denP.*ZZ5,2));
        sum_denP = gnpdf('sum',1,double(Z5));
        gnpdf('updatecluster',0,W1_prop,mu10',sigma10);
        gnpdf('updatecluster',1,1/K*ones(K,1),mu20',sigma20);
        gnpdf('z5',0,1,wwk,1);
        %denP_prop = sum(sum(denP_prop.*ZZ5,2));
        sum_denP_prop = gnpdf('sum',1,double(Z5));

        QQ = QQ+sum_denP_prop-sum_denP;
        if log(rand(1)) <=(QQ)
            U1(rr) =U1_prop(rr);
            W1 = W1_prop;
            accept_U1(rr) = 1;
        else
            U1_prop(rr) = U10(rr);
            %accept_U1(rr) = 0;
        end
    end
end



    
    
    
   
