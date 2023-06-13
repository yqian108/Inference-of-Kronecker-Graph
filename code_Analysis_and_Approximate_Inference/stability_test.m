clc;close all;clear;
addpath(genpath('./func/.'));
output_filename = 'StabilityTest.txt';

sh_lambda = 1e-2;
shuffle_prop =0.2;
m = 2;
vec_p = [0.3,0.4,0.5,0.6,0.7,0.8];
K = 10;
warm = 10;


for i = 1:6    
    p = vec_p(i);
    X = [-5, -8;-10, -18];
    X = X - mean( X(:) );
    x = X(:);
    x_true = x;
    N = m^K;
    P1 = p + X/sqrt(N);
    PK = generate_PK(P1, K);
    
    fid = fopen(output_filename,'a');
    fprintf(fid, "\n\n x_true':\t%s\n N:\t%s\t k:\t%s\t p:\t%s\t shuffle:\t%s\t\n lambda\t%s\n",num2str(x_true'),num2str(N), num2str(K),num2str(p),num2str(shuffle_prop),num2str(sh_lambda));
    fclose(fid);
    
    V_mse = zeros(1,warm);
    V_mse_nor = zeros(1,warm);
    for j = 1:warm    
        A = double(rand(N,N)<PK);       
        Pi_init_array = 1:N;
        [Pi_vector,A_shuffle] = shuffle(A,shuffle_prop, N, Pi_init_array); 
        
        bar_p = sum(A_shuffle(:))/N/N;
        p = bar_p^(1/K);
        Theta = generate_Theta(K,m,p);
        S_approx_shrink_shuffle = de_noise(A_shuffle, N, bar_p, (m-1)*K+1);
        x_init = zeros(m^2,1);
        hat_x = solve_convex_relaxation_func(S_approx_shrink_shuffle(:), Theta,N,x_init,sh_lambda, 20, 1e-8);
        V_mse_nor(j) = norm(x_true-hat_x,2)^2/norm(x_true,2)^2;
        V_mse(j) = norm(x_true-hat_x,2)^2;
    end
    mse_mean = mean(mean(V_mse));
    mse_std = std(V_mse);
    fid = fopen(output_filename,'a');
    fprintf(fid, "mse_nor:\t%s\n mse:\t%s\n mean:\t%s\n std:\t%s\n",num2str(V_mse_nor),num2str(V_mse),num2str(mse_mean),num2str(mse_std));
    fclose(fid);
end
