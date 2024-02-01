function [y_block,Theta_block] = get_block(S_approx_shrink,Theta,N,m,block_nums)
%     Random sampling for RNLA
%     Inputs:
%         -S_approx_shrink,: matrix of size N by N
%         -Theta: coefficient matrix of size N^2 by m^2
%         -N: number of nodes
%         -m: dimension of X
%         -block_nums: Number of sampled blocks
%     Outputs:
%         -y_block: vector of size block_nums * N
%         -Theta_block: matrix of size (block_nums * N) by m^2

    index = randperm(N, block_nums); 
    y_block = zeros(block_nums * N, 1);
    Theta_block = zeros(block_nums * N, m ^ 2);
    for i = 1 : block_nums
        y_block(((i-1)*N + 1):(i*N)) = S_approx_shrink(:,index(i));
        Theta_block(((i-1)*N + 1):(i*N),:) =  Theta(((index(i)-1)*N + 1):(index(i)*N),:);
    end
    
end