%     Using a permutation matrix to shuffle a graph
%     Inputs:
%         -A: adjacency of size N by N
%         -shuffle_prop: the shuffle proportion, in other words, the Hamming distance $d_H(pi,I) <= shuffle_ prop * N$
%         -N: number of nodes
%         -pi_init_array: vector of size N 
%     Outputs:
%         -Pi_vector: vector of size 1*N meets with Pi(i,pi_vector(i)) = 1
%         -A_shuffle: adjacency matrix of the shuffled graph

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

