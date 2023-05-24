function [S_approx_shrink] = de_noise(A, N, bar_p, c)
    % De-noise by constructing an estimator S_approx_shrink of SK
    % Input: A: adjacency matrix  c: top c singular values & sigma >= 2 sqrt(bar_p*(1-bar_p)) N: number of nodes
    % Output: the estimator S_approx_shrink of SK

    f_shrink = @(t) sqrt(t.^2 - 4*bar_p*(1-bar_p));
    bar_A = (A - ( sum( A(:)/N/N ) )*ones(N,N))/sqrt(N);   
    [U_bar_A, S_bar_A,V_bar_A]  = svds(bar_A,c);
    S_bar_A = diag(S_bar_A);
    S_approx_shrink = zeros(N);
    for temp = 1:c
        if(S_bar_A(temp) <= 2*sqrt(bar_p*(1-bar_p)))
            break;
        end
        S_approx_shrink = S_approx_shrink + f_shrink(S_bar_A(temp))*U_bar_A(:,temp)*V_bar_A(:,temp)';
    end

end


