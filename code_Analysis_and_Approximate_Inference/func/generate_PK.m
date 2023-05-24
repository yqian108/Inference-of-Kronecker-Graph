function PK = generate_PK(P1,K)
% generate_PK: generate probability matrix using K times Kronecker power
%   INPUT: P1 Kronecker initiator, K
%   OUTPUT: Kronecker probility matrix of size m^K by m^K
    tmp_K = K;
    tmp_P = P1;
    while tmp_K > 1
        tmp_P = kron(P1,tmp_P);
        %tmp_P = kron(tmp_P,P1);
        tmp_K = tmp_K - 1;
    end
    PK = tmp_P;
end