%% Test our model on real graph
clc;close all;clear;
addpath(genpath('./func/.'));
DATASET = 'REDDIT-BINARY';
input_edge = ['.\realgraph_datasets\',DATASET,'\',DATASET,'_A.txt'];
input_sets = ['.\realgraph_datasets\',DATASET,'\',DATASET,'_graph_indicator.txt'];
isdirect = 0;
lambda = 0.03;
m = 4;
output_file = ['.\realgraph_datasets\',DATASET,'\convexrelaxation\m',num2str(m),'.mat'];


% 1. ¶ÁÎÄ¼þ
[src,~,des] = textread(input_edge,'%d%s%d');  
graph_id = textread(input_sets,'%d');
[NodeNum,~] = size(graph_id);
[EdgeNum,~] = size(src);
graphNum = graph_id(NodeNum);


X = zeros(graphNum,m^2);

pre = 1;
st = 1;
ed = 0;
alnodes = 0;  
algraph = 0;  
edge_st = 1;   
i = 1;
while( algraph <= graphNum)  
    if(st > NodeNum)
        break;
    elseif (i <= NodeNum && graph_id(i) == graph_id(st))
        i = i + 1;
        continue;
    end

    ed = i - 1;
    nnodes = ed - st + 1;  
    k = ceil(log(nnodes)/log(m));
    N = m^k;
    
    A = zeros(N,N);
    max_nodesnum = ed;  
    algraph = algraph + 1;
    for j = edge_st:EdgeNum 
        if(src(j) > max_nodesnum | des(j) > max_nodesnum)
            edge_st = j;
            break;
        end
        
        node_src = src(j) - alnodes;
        node_des = des(j) - alnodes;
        
        A(node_src,node_des) = 1;
        if(~isdirect)
            A(node_des,node_src) = 1;
        end
    end
    
    
    bar_p = sum(A(:))/N/N;
    x_init = zeros(m^2,1);
    S_approx_shrink_shuffle = de_noise(A, N, bar_p,20);
    Theta = generate_Theta(k,m,bar_p);
    hat_x = solve_convex_relaxation_func(S_approx_shrink_shuffle(:), Theta,N,x_init,lambda, 20, 1e-8);

    X(algraph,:) = hat_x';

    st = i;
    alnodes = alnodes + nnodes;
end



save(output_file,'X','lambda','m');


