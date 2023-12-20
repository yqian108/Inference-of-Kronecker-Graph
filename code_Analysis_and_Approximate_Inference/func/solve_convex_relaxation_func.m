function [hat_x] = solve_convex_relaxation_func(y_block, Theta_block,N,x_init, lambda, max_iter, tolerance) 
    % Find the solution of min ||vec(S_approx_shrink) - Theta*x - sqrt(N)*e||_2^2 + lambda ||e||_1
    x_pre = x_init;
    para_lambda  = lambda/sqrt(N);
    obj_pre = 0;
    for i = 1:max_iter
        % step 1: argmin_e ||vec(S_approx_shrink) - Theta*x - sqrt(N)*e||_2^2 + lambda ||e||_1
        y = (y_block - Theta_block*x_pre)/sqrt(N);
        e = sign(y).*max(abs(y)-para_lambda /2,0);
      
        % step 2: argmin_x ||vec(S_approx_shrink) - Theta*x - sqrt(N)*e||_2^2 
        y = y_block(:) - sqrt(N)*e;
        x = (Theta_block'*Theta_block)^(-1)*Theta_block'*y;
       
        %obj = norm(x-x_pre,2)^2;
        obj = norm(y_block - Theta_block * x - sqrt(N) * e, 2)^2  + lambda * norm(e, 1);
        if(i > 1 && abs(obj - obj_pre)/obj_pre < tolerance)
            break;
        end
        x_pre = x;  
        obj_pre = obj;
    end
    hat_x = x;  
end
