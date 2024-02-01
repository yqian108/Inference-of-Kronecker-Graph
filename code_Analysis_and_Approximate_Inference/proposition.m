%% Prop 3.1 (i) & (iii)
clc;close all; clear;
addpath(genpath('./func/.'));


m = 2;
p = 0.9;
X = [0.5, 0; -1, 1];
x = X(:);

K_vec = 7:11;
error_output = zeros(length(K_vec),1);
rank_output = zeros(length(K_vec),1);

for i=1:length(K_vec)
    K = K_vec(i);
    N = m^K;
    
    P1 = p + X/sqrt(N);
    PK = generate_PK(P1, K);
    
    Theta = generate_Theta(K,m,p);
    PK_lin = p^K * ones(N,N) + reshape(Theta*x,[N,N])*sqrt(N);
    
    error_output(i) = norm(PK - PK_lin);
    rank_output(i) = rank(PK_lin);
end

figure
plot(m.^K_vec,error_output,'x-')
title('Operator norm error of linearization $P_K - P_K^{lin}$', 'Interpreter', 'latex')
xlabel('N', 'Interpreter', 'latex')

figure
semilogx(m.^K_vec,rank_output,'o-')
title('Rank of $P_K^{lin}$', 'Interpreter', 'latex')
xlabel('N', 'Interpreter', 'latex')

%% Prop 3.4: Signal-plus-noise approximation for centered adjacency bar_A
clc;close all; clear;
addpath(genpath('./func/.'));

K_vec = 7:12;
m = 2;
p = 0.9;
X = [0.5, 0; -1, 1];
 X = X - mean( X(:) );
x = X(:);


K_vec = 7:11;
error_output = zeros(length(K_vec),1);

for i=1:length(K_vec)
    K = K_vec(i);
    N = m^K;
    
    P1 = p + X/sqrt(N);
    PK = generate_PK(P1, K);
    
    Theta = generate_Theta(K,m,p);
    S = reshape(Theta*x,[N,N]);
    
    A = double(rand(N,N)<PK);
    bar_A = (A - ( sum( A(:)/N/N ) )*ones(N,N))/sqrt(N);
    bar_p = p^K;
    Z = normrnd(0,sqrt(bar_p*(1-bar_p)),N,N);
    
    error_output(i) = norm(bar_A - Z/sqrt(N) - S);
   
end

figure
plot(m.^K_vec,error_output,'x-')
title('Operator norm error of $\bar A - (Z/ \sqrt N + S_K)$', 'Interpreter', 'latex')
xlabel('N', 'Interpreter', 'latex')

