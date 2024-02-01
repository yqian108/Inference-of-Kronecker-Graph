function [y_block,Theta_block] = get_block(S_approx_shrink,Theta,N,m,block_nums)
%     Random sampling for RNLA
%     Input
%         -S_approx_shrink,: Matrix of size N*N
%         -Theta: Coefficient matrix of size N^2 * m^2
%         -N: Number of nodes
%         -m: Dimension of X
%         -block_nums: Number of sampled blocks
%     Output
%         -y_block: Vector of size block_nums * N
%         -Theta_block: Matrix of size (block_nums * N) * m^2

    index = randperm(N, block_nums); 
    y_block = zeros(block_nums * N, 1);
    Theta_block = zeros(block_nums * N, m ^ 2);
    for i = 1 : block_nums
        y_block(((i-1)*N + 1):(i*N)) = S_approx_shrink(:,index(i));
        Theta_block(((i-1)*N + 1):(i*N),:) =  Theta(((index(i)-1)*N + 1):(index(i)*N),:);
    end
    
end