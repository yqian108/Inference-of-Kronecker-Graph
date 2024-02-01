%% Replace soft thresholding with IHT(Single iteration & RNLA)
clc;close all;clear;
addpath(genpath('./func/.'));
output_filename = '.\IHT_RNLAMatlabTest_new.txt';

warm = 10;
blocks_num = 100;
q = 2;

top_k = 5;
time_str = num2str(16);
filename = ['.\synthetic_datasets\para',time_str,'.mat'];
load(filename);
%     fid = fopen(output_filename,'a');
%     fprintf(fid, "\n\n%s\n N:\t%s\t k:\t%s\t p:\t%s\t shuffle:\t%s\t\n", filename,num2str(N), num2str(K),num2str(p),num2str(shuffle_prop));
%     fprintf(fid, "x_true:\t%s\n top_k:\t%s\n", num2str(x_true'),num2str(top_k));
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
    hat_x =  solve_usingL0_single_block(S_approx_shrink_shuffle,Theta,N,m,blocks_num,50,top_k,1,1e-8); 
    mse_nor = norm(x_true-hat_x,2)^2/norm(x_true,2)^2;
    mse = norm(x_true-hat_x,2)^2;

    mse_list(j) = mse;

    fid = fopen(output_filename,'a');
    fprintf(fid, "iter:\t%s \n x:\t%s\n mse_nromalize:\t %s \n mse:\t %s \n",num2str(j), num2str(hat_x'),num2str(mse_nor),num2str(mse));
    fclose(fid);
end

t2=clock;
PF_Time = etime(t2,t1);

mse_mean = mean(mean(mse_list));
mse_std = std(mse_list);

%     fid = fopen(output_filename,'a');
%     fprintf(fid, "\n mse_mean:\t %s \t mse_std:\t%s\n",num2str(mse_mean), num2str(mse_std));
%     fprintf(fid, "\n time:\t %s \t second", num2str(PF_Time));
%     fclose(fid);
%     fprintf('\n\n TotalExeTm:     %f  second\n\n',PF_Time);



%% FUNCTIONS
function [x] = solve_usingL0_single_block(S_approx_shrink,Theta_original,N,m,block_nums,max_iter,hard_nums,hard_step,tolerance)
    % argmin ||v_S - Theta*x - d||_2^2  s.t. ||d||_0 <= 2kN
    
    [v_S, Theta] = get_block(S_approx_shrink,Theta_original,N,m,block_nums);
    
    obj_pre = 0;
    x = zeros(m^2,1);
    d = zeros(N * block_nums,1);
    for iter = 1:max_iter
        % step 1: d <- argmin ||y - d||_2^2  s.t. ||d||_0 <= 2kN
        y = v_S - Theta*x;    
        % step 1.1
        z = d - hard_step*(d-y);
        % step 1.2     
        [~,b] = sort(abs(z),'descend');
        d = zeros(N * block_nums,1);
        for index = 1:hard_nums
            d(b(index)) = z(b(index));
        end 
            
        % step 2: x <- argmin ||y - Theta*x||_2^2 
        y = v_S - d;
        x = (Theta'*Theta)^(-1)*Theta'*y;
       

        obj = norm(v_S - Theta * x -  d, 2)^2 ;  
        if(iter > 1 && abs(obj - obj_pre)/obj_pre < tolerance)
            fprintf('solve_L0:迭代%f次后收敛\n',iter);
            break;
        end
        obj_pre = obj;
    end

end