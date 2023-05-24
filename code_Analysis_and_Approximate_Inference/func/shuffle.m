function [Pi_vector,A_shuffle] = shuffle(A,shuffle_prop, N, Pi_init_array)
%     Get the shuffled matrix A_shuffle = Pi * A * Pi'
%     Input
%         -A: matrix of size N*N
%         -shuffle_prop: the Hamming distance d_H(pi,I) <= shuffle_prop*N
%         -pi_init_array: vector of size N
%             initial correspondence, can be regarded as identity matrix 
%     Output
%         -Pi_vector: vector of size 1*N meets with Pi(i,pi_vector(i)) = 1

    shuffle_nodes = round(N*shuffle_prop);
    
    % get Pi_vector
    Pi_vector = Pi_init_array;
    index = randperm(N);
    temp = Pi_vector(index(1:shuffle_nodes));
    Pi_vector(index(1:shuffle_nodes))=temp(randperm(shuffle_nodes));
    
    Pi_true = zeros(N,N);
    for i=1:N
        Pi_true(i,Pi_vector(i)) = 1;
    end
    A_shuffle = Pi_true * A * (Pi_true');
       
end

