function PK = generate_PK(P1,K)
%    Generate probability matrix using K times Kronecker power
%    Inputs:
%       -P1: Kronecker initiator
%       -K: iter times
%    Output: Kronecker probability matrix of size  m^K by m^K

    tmp_K = K;
    tmp_P = P1;
    while tmp_K > 1
        tmp_P = kron(P1,tmp_P);
        %tmp_P = kron(tmp_P,P1);
        tmp_K = tmp_K - 1;
    end
    PK = tmp_P;
end