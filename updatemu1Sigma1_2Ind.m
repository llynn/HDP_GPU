function [mu11 sigma11 accept_muSig11]=updatemu1Sigma1_2Ind(J,K,Z1,Z2,Z5,X1,lambda1,m1,delta1,Phi1,mu10,sigma10,SP,W11, mu20, sigma20, wwk)
    %%%%%%%%%%%%%%%% Update mu1 and Sigma1%%%%%%%%%%%%%%%%%%%%%%%%
    mu1_prop = mu10; 
    mu11 = mu10;
    Sigma1_prop =sigma10; 
    sigma11 = sigma10; 
    accept_muSig11 = zeros(J,1); 
     %ZZ5 =  full(sparse((1:n)',double(Z5),1,n,K));
        wwk(wwk<0.00001) = 0.00001;
        wwk = wwk./repmat(sum(wwk,2),1,K);
    for jj = 1:J/SP 
        for rr = 1+(jj-1)*SP:SP+(jj-1)*SP 
            ll1 = find(Z1==rr); nn1j = length(ll1);
            Xj1  = X1(ll1,:); xj1=Xj1';
            sumxj1 = sum(Xj1,1);
            ll2 = find(Z2==rr); nn2j = length(ll2);
            Xj2  = X1(ll2,:); xj2=Xj2';
            sumxj2 = sum(Xj2,1);
            mumu = (m1./lambda1 + sumxj1'+sumxj2')./(1/lambda1+nn1j+nn2j);
            mumu1 = (m1./lambda1 + sumxj1')./(1/lambda1+nn1j);
            mumu_Sig = (lambda1/(1+lambda1*(nn1j+nn2j))).*sigma10(:,:,rr);
            mumu_Sig1 = (lambda1/(1+lambda1*(nn1j))).*sigma10(:,:,rr);
            mu1_prop(:,rr) = mvnrnd(mumu, mumu_Sig); 
            df1 = delta1+nn1j+1;df = df1+nn2j; 
            tau1 = Phi1 + (mu1_prop(:,rr)-m1)*(mu1_prop(:,rr)-m1)'/lambda1 + (xj1-repmat(mu1_prop(:,rr),1,nn1j))*(xj1-repmat(mu1_prop(:,rr),1,nn1j))';
            tau = tau1+(xj2-repmat(mu1_prop(:,rr),1,nn2j))*(xj2-repmat(mu1_prop(:,rr),1,nn2j))';
            tau1 = (tau1+tau1')./2;
            tau = (tau+tau')./2;
            Sigma1_prop(:,:,rr) =  iwishrnd(tau,df);
        end
        qq = 1+(jj-1)*SP:SP+(jj-1)*SP;
        QQ = sum(logmvnormpdf(mu1_prop(:,qq),mumu1,mumu_Sig1)-logmvnormpdf(mu10(:,qq),mumu1,mumu_Sig1));
        QQ = QQ+ sum(logmvnormpdf(mu10(:,qq),mumu,mumu_Sig)-logmvnormpdf(mu1_prop(:,qq),mumu,mumu_Sig));
        QQ = QQ + sum(log_IWpdf(Sigma1_prop(:,:,qq),df1,tau1)-log_IWpdf(sigma10(:,:,qq),df1,tau1));
        QQ = QQ + sum(log_IWpdf(sigma10(:,:,qq),df,tau)-log_IWpdf(Sigma1_prop(:,:,qq),df,tau));
        gnpdf('updatecluster',0,W11,mu11',sigma11);
        gnpdf('updatecluster',1,1/K*ones(K,1),mu20',sigma20);
        %denP = gnpdf('z5',0,1,wwk,1);
        %denP = sum(denP.*ZZ5,2);
        gnpdf('z5',0,1,wwk,1);
        sum_denP = gnpdf('sum',1,double(Z5));
        gnpdf('updatecluster',0,W11,mu1_prop',Sigma1_prop);
        gnpdf('updatecluster',1,1/K*ones(K,1),mu20',sigma20);
        %denP_prop = gnpdf('z5',0,1,wwk,1);
        %denP_prop = sum(denP_prop.*ZZ5,2);
        gnpdf('z5',0,1,wwk,1);
        sum_denP_prop = gnpdf('sum',1,double(Z5));
        
        QQ = QQ+sum_denP_prop-sum_denP;
        if log(rand(1)) <= (QQ)
             mu11(:,qq) =mu1_prop(:,qq);
             sigma11(:,:,qq) = Sigma1_prop(:,:,qq);
             accept_muSig11(qq) = 1;
        else
             mu1_prop(:,qq) = mu10(:,qq);
             Sigma1_prop(:,:,qq) = sigma10(:,:,qq);
             %accept_muSig11(qq) = 0;
        end
    end
end

