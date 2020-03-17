function sigma_s = sanear(sigma)
[V, D] = eig(sigma);
D(D < 0) = 0.001;
tran = V';
sigma_s = V * D * tran;