function [mu1_prop sigma1_prop ]=updatemu1Sigma1_2IndG(J,Z1,Z2,X1,lambda1,m1,delta1,Phi1,mu10,sigma10)
    %%%%%%%%%%%%%%%% Update mu1 and Sigma1%%%%%%%%%%%%%%%%%%%%%%%%
    mu1_prop = mu10; 

    sigma1_prop =sigma10; 

    
    for rr = 1:J
        ll1 = find(Z1==rr); nn1j = length(ll1);
        Xj1  = X1(find(Z1==rr),:); xj1=Xj1';
        sumxj1 = sum(Xj1,1);
        ll4 = find(Z2==rr); nn4j = length(ll4);
        Xj4  = X1(find(Z2==rr),:); xj4=Xj4';
        sumxj4 = sum(Xj4,1);
        mumu = (m1./lambda1 + sumxj1'+sumxj4')./(1/lambda1+nn1j+nn4j);
        mu1_prop(:,rr) = mvnrnd(mumu, (lambda1/(1+lambda1*(nn1j+nn4j))).*sigma10(:,:,rr)); 
        df = delta1+nn1j+nn4j+1;
        tau =Phi1 + (mu1_prop(:,rr)-m1)*(mu1_prop(:,rr)-m1)'/lambda1 + (xj1-repmat(mu1_prop(:,rr),1,nn1j))*(xj1-repmat(mu1_prop(:,rr),1,nn1j))'...
            +(xj4-repmat(mu1_prop(:,rr),1,nn4j))*(xj4-repmat(mu1_prop(:,rr),1,nn4j))';
        sigma1_prop(:,:,rr) =  iwishrnd((tau+tau')./2,df);
    end
       
end