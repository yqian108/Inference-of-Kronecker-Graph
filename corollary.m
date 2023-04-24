%% Corollary: Shrinkage estimation of small-rank S_K
clc;close all; clear;
addpath(genpath('./func/.'));

m = 2;
p = 0.9;
X = [0.5, 0; -1, 1];
K_vec = 7:11;
% X = [1, 0; -1, 2];
% X = [1,0,2;-1,2,3;0,1,2];
% X = [3, 2.5; 2, 1];
x = X(:);

% error_output = zeros(length(K_vec),1);

for i=1:length(K_vec)
    K = K_vec(i);
    N = m^K;
    bar_p = p^K;
    f_shrink = @(t) sqrt(t.^2 - 4*bar_p*(1-bar_p));
    
    P1 = p + X/sqrt(N)
    PK = generate_PK(P1, K);
    
    Theta = generate_Theta(K,m,p);
    S = reshape(Theta*x,[N,N]);
    [U_S,Ell_S,V_S] = svds(S/sqrt(bar_p*(1-bar_p)));
    ell = diag(Ell_S);
    
    A = double(rand(N,N)<PK);
    bar_A = (A - ( sum( A(:)/N/N ) )*ones(N,N))/sqrt(N);
    [U_bar_A, S_bar_A, V_bar_A] = svd(bar_A);
    S_bar_A = diag(S_bar_A);
    
    temp = 1;
    S_approx_shrink = zeros(N);
    S_approx = zeros(N);
    while ell(temp) > 1
        S_approx_shrink = S_approx_shrink + f_shrink(S_bar_A(temp))*U_bar_A(:,temp)*V_bar_A(:,temp)';
        S_approx = S_approx + S_bar_A(temp)*U_bar_A(:,temp)*V_bar_A(:,temp)';
        temp = temp + 1;
    end
    
    
%     error_output(i) = norm(S-S_approx_shrink,'fro')^2;
    
    disp('=========Shrinkage estimation======')
    disp( ['S_approx-Empirical: ', num2str( norm(S-S_approx,'fro')^2 )] )
    disp( ['S_approx_shrink-Empirical: ', num2str( norm(S-S_approx_shrink,'fro')^2 )] )
    disp( ['Theory: ', num2str( 2*rank(S)*bar_p*(1-bar_p) )] )
   
end





