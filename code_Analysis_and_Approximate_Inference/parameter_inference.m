clc;close all;clear;
% output_filename = './test.txt';
addpath(genpath('./func/.'));
warm = 10;

time_str = num2str(16);
filename = ['.\synthetic_datasets\para',time_str,'.mat'];
% load(filename);
% fid = fopen(output_filename,'a');
% fprintf(fid, "\n\n%s\n N:\t%s\t k:\t%s\t p:\t%s\t shuffle:\t%s\t\n", filename,num2str(N), num2str(K),num2str(p),num2str(shuffle_prop));
% fprintf(fid, "x_true:\t%s\n", num2str(x_true'));
% fclose(fid);

t1 = clock;
for j = 1:warm
    load(filename);
    bar_p = p^K;
    Theta = generate_Theta(K,m,p);
    S_approx_shrink_shuffle = de_noise(A_shuffle, N, bar_p, 6);
    x_init = zeros(m^2,1);
    hat_x = solve_convex_relaxation_func(S_approx_shrink_shuffle(:), Theta,N,x_init,0.003, 20, 1e-8);
    mse = norm(x_true-hat_x,2)^2;
%     fid = fopen(output_filename,'a');
%     fprintf(fid, "iter:\t%s \n x:\t%s\n mse:\t %s \n",num2str(j), num2str(hat_x'),num2str(mse));
%     fclose(fid);
end

t2=clock;
PF_Time = etime(t2,t1);
% fid = fopen(output_filename,'a');
% fprintf(fid, "\n time:\t %s \t second", num2str(PF_Time));
% fclose(fid);
fprintf('\n\n TotalExeTm£º     %f  second\n\n',PF_Time);













