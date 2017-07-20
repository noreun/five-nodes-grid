function I = MI_time(Cov_X,Cov_XY,Cov_Y)

% conditional covariance matrix
Cov_X_Y = Cov_cond(Cov_X,Cov_XY,Cov_Y);                 %%% <= important!

% conditional entropy in the whole system
H_cond = H_gauss(Cov_X_Y);        

% checking that finite and real entropy was obtained
if isinf(H_cond) == 1
    error('Alert: Infinity Entropy\n')
end
if isreal(H_cond) == 0
    error('Alert: Complex Entropy\n')
end

% mutual information
I = H_gauss(Cov_X) - H_cond;
 
end
