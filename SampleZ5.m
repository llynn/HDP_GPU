function Z5 = SampleZ5(mu1,sigma1,mu2,sigma2,W1,wwk)
[J K ] = size(wwk);
gnpdf('updatecluster',0,W1,mu1,sigma1);
gnpdf('updatecluster',1,1/K*ones(K,1),mu2,sigma2);
Z5 = gnpdf('z5',0,1,wwk);
