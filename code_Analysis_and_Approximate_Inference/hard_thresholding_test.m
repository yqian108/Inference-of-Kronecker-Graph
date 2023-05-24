%% Replace soft thresholding with IHT
% argmin ||vec(S_k) - Theta*x - \sqrt(N) d||_2^2 + lambda ||d||_0

clc;close all;clear;
addpath(genpath('./func/.'));
output_filename = '.\IHT_test.txt';

warm = 10;

top_k = 5;
time_str = num2str(16);
filename = ['.\synthetic_datasets\para',time_str,'.mat'];
load(filename);
% fid = fopen(output_filename,'a');
% fprintf(fid, "\n\n%s\n N:\t%s\t k:\t%s\t p:\t%s\t shuffle:\t%s\t\n", filename,num2str(N), num2str(K),num2str(p),num2str(shuffle_prop));
% fprintf(fid, "x_true:\t%s\n top_k:\t%s\n", num2str(x_true'),num2str(top_k));
% fclose(fid);
%     
t1 = clock;
for j = 1:warm
    load(filename);
    bar_p = p^K;
    Theta = generate_Theta(K,m,p);
    S_approx_shrink_shuffle = de_noise(A_shuffle, N, bar_p, 6);
    hat_x =  solve_usingL0(S_approx_shrink_shuffle(:),Theta,N,m,50,150,top_k,1,1e-8); 
    mse_nor = norm(x_true-hat_x,2)^2/norm(x_true,2)^2;
    mse = norm(x_true-hat_x,2)^2;

%     fid = fopen(output_filename,'a');
%     fprintf(fid, "iter:\t%s \n x:\t%s\n mse_nromalize:\t %s \n mse:\t %s \n",num2str(j), num2str(hat_x'),num2str(mse_nor),num2str(mse));
%     fclose(fid);
end

t2=clock;
PF_Time = etime(t2,t1);
% fid = fopen(output_filename,'a');
% fprintf(fid, "\n time:\t %s \t second", num2str(PF_Time));
% fclose(fid);
fprintf('\n\n TotalExeTm：     %f  second\n\n',PF_Time);







%% FUNCTIONS
function [x] = solve_usingL0(v_S,Theta,N,m,max_iter,hard_max_iter,hard_nums,hard_step,tolerance)
%     Find the solution of min ||v_S - Theta*x - sqrt(N) d||_2^2  s.t. ||d||_0 <= 2kN
%     Input
%         -v_S: vector of size N^2 * 1
%         -Theta:matrix of size N^2 * m*2
%         -hard_max_iter, hard_nums,hard_step: parameters for solving IHT
%     Output: the solution of size m^2 * 1
    obj_pre = 0;
    x = zeros(m^2,1);
    for iter = 1:max_iter
        % step 1: d <- argmin ||y - \sqrt(N) d||_2^2  s.t. ||d||_0 <= 2kN
        y = (v_S - Theta*x)/sqrt(N);
        d = IHT(y,N,hard_nums,hard_step,hard_max_iter,tolerance);  % 返回nums： 非0元素个数
        d(1:10);
        % step 2: x <- argmin ||y - Theta*x||_2^2 
        y = v_S - sqrt(N)*d;
        x = (Theta'*Theta)^(-1)*Theta'*y;
       
        %obj = norm(x-x_pre,2)^2;
        obj = norm(v_S - Theta * x - sqrt(N) * d, 2)^2 ;  
        if(iter > 1 && abs(obj - obj_pre)/obj_pre < tolerance)
            break;
        end
        obj_pre = obj;
    end



end
function [d] = IHT(y,N,hard_nums,hard_step,max_iter,tolerance)
    % IHT: find the solution of min ||y-d||^2  s.t. ||d||_0 <= 2kN  
    % Input 
    %     -y: vector of size N^2 * 1
    %     -hard_nums: the number of choosen values after sorting 
    %     -hard_step: step size of gradient descent
    % Output
    %     -d:the solution of size N^2 * 1
    obj_pre = 0;
    d = zeros(N^2,1);
    for iter = 1:max_iter
        z = d - hard_step*(d-y);
        [~,b] = sort(abs(z),'descend');
        d = zeros(N^2,1);
        for index = 1:hard_nums
            if(z(b(index)) == 0)
                break;
            end
            d(b(index)) = z(b(index));
        end
        
        obj = norm(y - d, 2)^2 ;  
        if(iter > 1 && abs(obj - obj_pre)/obj_pre < tolerance)
            break;
        end
        obj_pre = obj;
       
    end
end


