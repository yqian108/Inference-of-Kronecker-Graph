function [Theta] = generate_Theta(k,m,p)
%    Generate coefficient matrix
%    Inputs: 
%       -K: iter times
%       -m: Kronecker initiator size
%       -p: parameter of P1
%    Output: coefficient matrix of size m^2k by m^2k

N = m^k;         
Theta = zeros(N^2,m^2);

for i = 1:m
    for j = 1:m
        Z = zeros(m);
        Z(i,j) = 1;
        temp = Z;
        for q = 2:k
            temp = kron(temp, ones(m)) + kron(ones(m^(q-1)), Z);
        end
        Theta(:,m*(j-1)+i) = temp(:);
    end
end

Theta = Theta/N;
Theta = p^(k-1)*Theta;
end