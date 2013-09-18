function [V1 WW1 accept_V] = updateVWWG_2Ind(K,J,gamma20,Vk0,alpha2,V0,WW0,SPV,mu20, sigma20)
   gnpdf('updatecluster',1,WW0,mu20',sigma20);
   Z3 = gnpdf('pdf&sample',1);
   [V_prop WW_prop aVV bVV] = updateVWWG(Z3,K,gamma20);

   V1 = V0; V1_prop = V0;accept_V= zeros(K,1); WW1 = WW0; WW1_prop = WW0;
    %%%%%%%%%%%%%%%% Update V %%%%%%%%%%%%%%%%%%%%%%%%
    for jj = 1:(K-1)/SPV
        rr = 1+(jj-1)*SPV:SPV+(jj-1)*SPV; 
        V1_prop(rr) = V_prop(rr);
        WW1_prop(1) = V1_prop(1); prod = 1-V1_prop(1);    
        for rrr = 2:K-1
            WW1_prop(rrr) = prod*V1_prop(rrr);
            prod = prod*(1-V1_prop(rrr));
        end
        WW1_prop(K) = prod;
        
        log_post = sum(betapdf_log(V0(rr),1,gamma20));
        log_post_prop = sum(betapdf_log(V1_prop(rr),1,gamma20));
        for j = 1:J
            log_post = log_post+sum(betapdf_log(Vk0(j,:)', alpha2*WW1(1:K-1), alpha2*(1-cumsum(WW1(1:K-1)))));
            log_post_prop = log_post_prop + sum(betapdf_log(Vk0(j,:)', alpha2*WW1_prop(1:K-1), alpha2*(1-cumsum(WW1_prop(1:K-1)))));
        end
        
        if log(rand(1))<log_post_prop-log_post+sum(betapdf_log(V0(rr),aVV(rr),bVV(rr))-betapdf_log(V1_prop(rr),aVV(rr),bVV(rr)))
            accept_V(rr) = 1;
            WW1 = WW1_prop;
            V1 = V1_prop;
        else
         
            V1_prop(rr) = V0(rr);
        end
    
    end
    
  

    