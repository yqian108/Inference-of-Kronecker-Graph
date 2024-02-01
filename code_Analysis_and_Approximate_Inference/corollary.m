%% Corollary C.3: Shrinkage estimation of small-rank S_K
clc;close all; clear;
addpath(genpath('./func/.'));

m = 2;
p = 0.9;
X = [-0.5, 1; -1, 2];
X = X - mean( X(:) );
K_vec = 7:11;
x = X(:);

% error_output = zeros(length(K_vec),1);
plot_x = zeros(1,length(K_vec));
plot_y_theory = zeros(1,length(K_vec));
plot_y_empirical = zeros(1,length(K_vec));

% to be done
for i=1:length(K_vec)
    K = K_vec(i);
    N = m^K;
    bar_p = p^K;
    f_shrink = @(t) sqrt(t.^2 - 4*bar_p*(1-bar_p));
    
    P1 = p + X/sqrt(N);
    PK = generate_PK(P1, K);
    
    Theta = generate_Theta(K,m,p);
    S = reshape(Theta*x,[N,N]);
    [U_S,Ell_S,V_S] = svd(S/sqrt(bar_p*(1-bar_p)));
    ell = diag(Ell_S);
    disp( ['Singular values of S/sqrt(bar_p*(1-bar_p)): ', num2str((ell(1:6))')]);
    
    A = double(rand(N,N)<PK);
    bar_A = (A - ( sum( A(:)/N/N ) )*ones(N,N))/sqrt(N);
    [U_bar_A, S_bar_A, V_bar_A] = svd(bar_A);
    S_bar_A = diag(S_bar_A);
    
    temp = 1;
    S_approx_shrink = zeros(N);
    S_approx = zeros(N);
    while S_bar_A(temp) > 2*sqrt(bar_p*(1-bar_p))
        S_approx_shrink = S_approx_shrink + f_shrink(S_bar_A(temp))*U_bar_A(:,temp)*V_bar_A(:,temp)';
        % S_approx = S_approx + S_bar_A(temp)*U_bar_A(:,temp)*V_bar_A(:,temp)';
        temp = temp + 1;
    end

    theory = 0;
    for temp = 1:rank(S)
        sigma = ell(temp)* sqrt(bar_p*(1-bar_p));
        if sigma > 1
            theory = theory + bar_p*(1-bar_p)*(2 - bar_p*(1-bar_p)*sigma^(-2));
        else
            theory = theory + sigma^2;
        end
    end
    
%     error_output(i) = norm(S-S_approx_shrink,'fro')^2;
    
    disp('=========Shrinkage estimation======')
    % disp( ['S_approx-Empirical: ', num2str( norm(S-S_approx,'fro')^2 )] )
    disp( ['S_approx_shrink-Empirical: ', num2str( norm(S-S_approx_shrink,'fro')^2 )] )
    disp( ['Theory: ', num2str(theory)])
    
    plot_x(i) = N;
    plot_y_theory(i) = theory;
    plot_y_empirical(i) =  norm(S-S_approx_shrink,'fro')^2;
   
end

figure
plot(plot_x,plot_y_theory,'-xr','linewidth',1);
hold on
plot(plot_x,plot_y_empirical,'-xb','linewidth',1);
legend("Theory","Empirical");





