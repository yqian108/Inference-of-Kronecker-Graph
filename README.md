# Inference-of-Kronecker-Graph
This repository contains code to reproduces the results in the paper "*Analysis and Approximate Inference of
Large and Dense Random Kronecker Graph*".
## About the code
* `code_Analysis_and_Approximate Inference` contains 
  * **theorem.m**: test for Theorem 2
  * **proposition.m**: test for Proposition 1, 2
  * **corolarry.m**: test for corolarry 1
  * **generate_data.m**: generate a shuffled graph under our model
  * **parameter_inference.m**: the approach proposed (Algorithm 1- Algorithm 3)
  * **realgraph_test.m**: applying our model on realistic graphs
  * **hard_thresholding_test.m**: using Iterative Hard-thresholding (IHT) rather than soft thresholding in Algorithm 3
  * `func` contains 
    * **generate_PK.m** : generate probability matrix using K times Kronecker power
      * Input:
        * -P1: Kronecker initiator
        * -K: Iter times
      * Output: Kronecker probability matrix of size $m^K \times m^K$
    * **generate_Theta.m**: generate coefficient matrix
      * Input: 
        * -K: Iter times
        * -m: Kronecker initiator size
        * -p: Parameter of P1
      * Output: coefficient matrix of size $m^{2K} \times m^2$ 
    * **shuffle.m**: using a permutation matrix to shuffle a graph
      * Input: 
          * -A: Adjacency matrix of size $N \times N$
          * -shuffle_prop: The shuffle proportion, in other words, the Hamming distance $d_H(pi,I) <= shuffle\\_ prop * N$
          * -N: The number of nodes
          * -pi_init_array: Vector of size $N$
       * Output:
          * -Pi_vector: Vector of size $1 \times N$ meets with $Pi(i,Pi\\_ vector(i)) = 1$
          * -A_shuffle: The adjacency matrix of the shuffled graph
     * **de_noise.m**: de-noise by constructing an estimator S_approx_shrink of SK
       * Input:
         * -A:  Adjacency matrix of size $N \times N$
         * -N: The number of ndoes
         * -bar_p: Constant between $0$ and $1$
         * -c: The number of choosen singular values
       * Output: an estimator S_approx_shrink of $N$ by $N$
     * **solve_convex_relaxation_func.m**: the implement of Algorithm 3
        * Input: 
          * -y_block: Vector of size $N^2$
          * -Theta_block: Coefficient matrix of $N^2 \times m^2$
          * -N: The number of nodes
          * -x_init: Initial value of x
          * -lambda: The hyperparameter in soft thresholding
          * -max_iter: Maximum number of iterations
          * -tolerance: Condition for exiting an iteration
        * Output: vector of size $m^2$
     
* `classification_task` contains 
  * **main.py**: test for classification_task, following the experiments in [^1]
  * **classes.py**: contains several modules for performing classification task in **main.py**
    
  
## Dependencies
To execute the code in `classification_task`, you can install the follwing basic packages by yourself:
* [Python](https://www.python.org/): tested with version 3.7.10
* [Pytorch](https://pytorch.org/): tested with version 1.8.1
* [Pandas](https://pandas.pydata.org/): tested with version 1.3.5
* [Scikit-learn](https://scikit-learn.org/stable/): tested with version 1.0.2



## References
[^1]:  Errica F, Podda M, Bacciu D, et al. A fair comparison of graph neural networks for graph classification[J]. arXiv preprint arXiv:1912.09893, 2019.
