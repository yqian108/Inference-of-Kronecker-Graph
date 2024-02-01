function [S_approx_shrink] = de_noise_rsvd(A, N, bar_p, c, q)
%     De-noise by constructing an estimator S_approx_shrink of SK using rsvd
%     Inputs:
%           -A: adjacency matrix  
%           -bar_p: constant between 0-1
%           -c: number of choosen singular values 
%           -N: number of nodes
%           -q: number of iterations
%     Output: an estimator S_approx_shrink of N by N (using rsvd)

    f_shrink = @(t) sqrt(t.^2 - 4*bar_p*(1-bar_p));
    bar_A = (A - ( sum( A(:)/N/N ) )*ones(N,N))/sqrt(N);  
    
    % step 1
%     l = min(c + oversampling, N);
    l = 2*c;
    Omega = randn(N, l);
    
%     Y = bar_A * Omega;
%     Q = orth(Y);   
    
    Y0 = bar_A * Omega;
    [Q, ~] = qr(Y0, 0);
    for index = 1:q
        Y = bar_A' * Q;
        [Q,~] = qr(Y, 0);
        Y = bar_A * Q;
        [Q,~] = qr(Y, 0);
    end
 
    % step 2
    B = Q' * bar_A;
    [U, S_bar_A, V_bar_A] = svd(B, 'econ');
    S_bar_A = diag(S_bar_A);
    U_bar_A =  Q * U;
    
    S_approx_shrink = zeros(N);
    for temp = 1:c
        if(S_bar_A(temp) <= 2*sqrt(bar_p*(1-bar_p)))
            break;
        end
        S_approx_shrink = S_approx_shrink + f_shrink(S_bar_A(temp))*U_bar_A(:,temp)*V_bar_A(:,temp)';
    end

end