%% Using convex relaxation and RNLA for parameter inference
clc;close all;clear;
addpath(genpath('./func/.'));
output_filename = '.\RelaxationBlockMatlabTest_new.txt';
sh_lambda = 1e-2;
warm = 10;
blocks_num = 100;
q = 2;

time_str = num2str(16);
filename = ['.\synthetic_datasets\para',time_str,'.mat'];
load(filename);
%     fid = fopen(output_filename,'a');
%     fprintf(fid, "\n\n%s\n N:\t%s\t k:\t%s\t p:\t%s\t shuffle:\t%s\t\n lambda\t%s\n", filename,num2str(N), num2str(K),num2str(p),num2str(shuffle_prop),num2str(sh_lambda));
%     fprintf(fid, "oversampling: \t%s\t block_num: \t%s\t x_true:\t%s\n",num2str(q), num2str(blocks_num), num2str(x_true'));
%     fclose(fid);
mse_list = zeros(1,warm);
t1 = clock;
for j = 1:warm    
    load(filename);
    bar_p = sum(A_shuffle(:))/N/N;
    p = bar_p^(1/K);
    Theta = generate_Theta(K,m,p);
    S_approx_shrink_shuffle = de_noise_rsvd(A_shuffle, N, bar_p, (m-1)*K+1, q);
    x_init = zeros(m^2,1);
    hat_x = solve_convex_relaxation_func_block(S_approx_shrink_shuffle, Theta,blocks_num, N, m,x_init,sh_lambda, 20, 1e-8);

    mse_nor = norm(x_true-hat_x,2)^2/norm(x_true,2)^2;
    mse = norm(x_true-hat_x,2)^2;

    mse_list(j) = mse;

%         fid = fopen(output_filename,'a');
%         fprintf(fid, "iter:\t%s \n x:\t%s\n mse_nor:\t %s \n mse:\t%s\n",num2str(j), num2str(hat_x'),num2str(mse_nor),num2str(mse));
%         fclose(fid);
end

t2=clock;
PF_Time = etime(t2,t1);

mse_mean = mean(mean(mse_list));
mse_std = std(mse_list);

%     fid = fopen(output_filename,'a');
%     fprintf(fid, "\n mse_mean:\t %s \t mse_std:\t%s\n",num2str(mse_mean), num2str(mse_std));
%     fprintf(fid, "\n time:\t %s \t second", num2str(PF_Time));
%     fclose(fid);
%     fprintf('\n\n TotalExeTm£º     %f  second\n\n',PF_Time);



%% Functions
function [hat_x] = solve_convex_relaxation_func_block(S_approx_shrink, Theta, block_nums, N, m, x_init, lambda, max_iter, tolerance) 

    [y_block, Theta_block] = get_block(S_approx_shrink,Theta,N,m,block_nums);
    
    
    % Find the solution of min ||vec(S_approx_shrink) - Theta*x - sqrt(N)*e||_2^2 + lambda ||e||_1
    x_pre = x_init;
    para_lambda  = lambda/sqrt(N);
    obj_pre = 0;
    for i = 1:max_iter
        % step 1: argmin_e ||vec(S_approx_shrink) - Theta*x - sqrt(N)*e||_2^2 + lambda ||e||_1
        y = (y_block - Theta_block*x_pre)/sqrt(N);
        e = sign(y).*max(abs(y)-para_lambda /2,0);
      
        % step 2: argmin_x ||vec(S_approx_shrink) - Theta*x - sqrt(N)*e||_2^2 
        y = y_block(:) - sqrt(N)*e;
        x = (Theta_block'*Theta_block)^(-1)*Theta_block'*y;
       
        %obj = norm(x-x_pre,2)^2;
        obj = norm(y_block - Theta_block * x - sqrt(N) * e, 2)^2  + lambda * norm(e, 1);
        if(i > 1 && abs(obj - obj_pre)/obj_pre < tolerance)
            break;
        end
        x_pre = x;  
        obj_pre = obj;
    end
    hat_x = x;  
end

