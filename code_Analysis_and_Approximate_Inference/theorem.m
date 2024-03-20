%% Theorem C.1: Asymptotic characterization of adjacency singular values
clc;close all; clear;
addpath(genpath('./func/.'));


m = 2;
K = 10;
p = 0.8;
X = [-3, -1;3, 2];
X = X - mean( X(:) );
x = X(:);


bar_p = p^K;
N = m^K;

P1 = p + X/sqrt(N)
PK = generate_PK(P1, K);
    
Theta = generate_Theta(K,m,p);
S = reshape(Theta*x,[N,N]);
[U_S,Ell_S,V_S] = svd(S/sqrt(bar_p*(1-bar_p)));
ell = diag(Ell_S);


A = double(rand(N,N)<PK);
bar_A = (A - ( sum( A(:)/N/N ) )*ones(N,N))/sqrt(N);
[U_bar_A, S_bar_A, V_bar_A] = svd(bar_A);
S_bar_A = diag(S_bar_A);


edges=linspace(+eps,2*sqrt(bar_p*(1-bar_p))-eps,60);
mu = sqrt( 4*bar_p*(1-bar_p) - edges.^2)/(bar_p)/(1-bar_p)/pi;


figure
histogram(S_bar_A, 50, 'Normalization', 'pdf', 'EdgeColor', 'white');
title('Singularvalue behavior')
hold on
plot(edges,mu,'r', 'Linewidth',2);


for i = 1:N
    if ell(i)>1
        spike_approx = sqrt(bar_p*(1-bar_p))*sqrt(2 + ell(i)^2 + ell(i)^(-2));
        plot(spike_approx,0,'xr')
    else
        break;
    end
end

for i = 1:N
    if ell(i)>1
        vi_bar_A = V_bar_A(:,i);
        vi_S = V_S(:,i);
        ui_bar_A = U_bar_A(:,i);
        ui_S = U_S(:,i);
        
        disp('=========Singular alignment======')
        disp( ['Empirical: ', num2str( (vi_S'*vi_bar_A)^2 )] )
        disp( ['Empirical: ', num2str( (ui_S'*ui_bar_A)^2 )] )
        disp( ['Theory: ', num2str( 1 - ell(i)^(-2) )] )
    else
        break;
    end
end

