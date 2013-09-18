function [Z1 Z2] = updateZ1Z2_2Ind(W10,mu10,sigma10,WWk0,mu20,sigma20,Z5)
    [J K ] = size(WWk0);
    gnpdf('updatecluster',0,W10,mu10',sigma10);
    gnpdf('updatecluster',1,1/K*ones(K,1),mu20',sigma20);   %ddensity2
    %gnpdf('zz',0,1,WWk0,double(Z5));
    [Z1 Z2 ] = gnpdf('zz',0,1,WWk0,double(Z5),0);    
   % gnpdf('updatecluster',1,WW0,mu20',sigma20);
   % Z3 = gnpdf('pdf&sample',1);
end