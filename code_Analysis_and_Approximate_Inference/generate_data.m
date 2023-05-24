%% Generate a synthetic graph
clc;close all;clear;
addpath(genpath('./func/.'));
shuffle_prop =0.2;
m = 2;
K = 10;
p = 0.9;
X = [-3, -1;3, 2];
X = X - mean( X(:) );
x = X(:);
x_true = x;


bar_p = p^K;
sqrt(bar_p*(1-bar_p))
N = m^K;

P1 = p + X/sqrt(N);
PK = generate_PK(P1, K);
A = double(rand(N,N)<PK);
Theta = generate_Theta(K,m,p);
S = reshape(Theta*x,[N,N]);

% shuffle
Pi_init_array = 1:N;
[Pi_vector,A_shuffle] = shuffle(A,shuffle_prop, N, Pi_init_array); 


save(".\synthetic_graph.mat","A_shuffle" ,"K", "N","p","x_true","m","shuffle_prop");


