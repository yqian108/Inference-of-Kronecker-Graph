# Inference-of-Kronecker-Graph
This repository contains code to reproduces the results in the paper "*Analysis and Approximate Inference of
Large Random Kronecker Graph*".
## About the code
* `code_Analysis_and_Approximate_Inference` contains 
  * **theorem.m**: test for Theorem C.1
  * **proposition.m**: test for Proposition 3.1, 3.4
  * **corolarry.m**: test for corolarry C.3
  * **generate_data.m**: generate a shuffled graph under our model
  * **parameter_inference_relax.m**: the approach proposed (Algorithm 1- Algorithm 3) using convex relaxation [^4]
  * **parameter_inference_IHT.m**: the approach proposed (Algorithm 1- Algorithm 3) using Iterative Hard-thresholding (IHT) [^2][^3]
  * **parameter_inference_relax_RNLA.m**: the approach proposed (Algorithm 1- Algorithm 3) using convex relaxation with RNLA acceleration [^5]
  * **parameter_inference_IHT_RNLA.m**: the approach proposed (Algorithm 1- Algorithm 3) using IHT with RNLA acceleration [^5]
  * **stability_test.m**: test for the stability of our approach proposed (Algorithm 1- Algorithm 3)
  * **realgraph_test.m**: applying our model on realistic graphs
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
          * -N: Number of nodes
          * -pi_init_array: Vector of size $N$
       * Output:
          * -Pi_vector: Vector of size $1 \times N$ meets with $Pi(i,Pi\\_ vector(i)) = 1$
          * -A_shuffle: The adjacency matrix of the shuffled graph
     * **de_noise.m**: de-noise by constructing an estimator S_approx_shrink of SK
       * Input:
         * -A:  Adjacency matrix of size $N \times N$
         * -N: Number of ndoes
         * -bar_p: Constant between $0$ and $1$
         * -c: Number of choosen singular values
       * Output: an estimator S_approx_shrink of $N$ by $N$
     * **de_noise_rsvd.m**: de-noise by constructing an estimator S_approx_shrink of SK using rsvd
       * Input:
         * -A:  Adjacency matrix of size $N \times N$
         * -N: Number of ndoes
         * -bar_p: Constant between $0$ and $1$
         * -c: Number of choosen singular values
         * -q: Number of iterations
       * Output: an estimator S_approx_shrink of $N$ by $N$ by rsvd
     * **get_block.m**: random sampling for RNLA 
       * Input:
         * -S_approx_shrink:  Matrix of size $N$ by $N$
         * -Theta: Coefficient matrix of size $N^2 \times m^2$
         * -N: Number of ndoes
         * -m: Kronecker initiator size
         * -block_nums: Number of sampled blocks
       * Output:
         * -y_block: Vector of size $block\\_ nums * N $
         * -Theta_block: Matrix of size $block\\_ nums * N \times m^2$
     * **solve_convex_relaxation_func.m**: the implement of Algorithm 3
        * Input: 
          * -y_block: Vector of size $N^2$
          * -Theta_block: Coefficient matrix of $N^2 \times m^2$
          * -N: Number of nodes
          * -x_init: Initial value of x
          * -lambda: Hyperparameter in soft thresholding
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

## Contact information
* Zhenyu Liao
  * Assistant Professor at EIC, Huazhong University of Science and Tech
  * Website: [https://zhenyu-liao.github.io/](https://zhenyu-liao.github.io/)
  * E-mail: [zhenyu_liao@hust.edu.cn](mailto:zhenyu_liao@hust.edu.cn)

* Yuanqian Xia
  * Master at EIC, Huazhong University of Science and Tech
  * E-mail: [m202172379@hust.edu.cn](mailto:m202172379@hust.edu.cn)

* Chengmei Niu
  * PhD student at EIC, Huazhong University of Science and Tech
  * E-mail: [chengmeiniu@hust.edu.cn](mailto:chengmeiniu@hust.edu.cn)

* Yong Xiao
  * Full Professor at EIC, Huazhong University of Science and Tech
  * Website: [http://eic.hust.edu.cn/professor/xiaoyong/index.html](http://eic.hust.edu.cn/professor/xiaoyong/index.html)
  * E-mail: [yongxiao@hust.edu.cn](mailto:yongxiao@hust.edu.cn)

## References
[^1]: Errica F, Podda M, Bacciu D, et al. A Fair Comparison of Graph Neural Networks for Graph Classification[C]//International Conference on Learning Representations. 2019.
[^2]: Blumensath T, Davies M E. Iterative thresholding for sparse approximations[J]. Journal of Fourier analysis and Applications, 2008, 14: 629-654.
[^3]: Jain P, Kar P. Non-convex optimization for machine learning[J]. Foundations and Trends® in Machine Learning, 2017, 10(3-4): 142-363.
[^4]: Slawski M, Ben-David E. Linear regression with sparsely permuted data[J]. Electronic Journal of Statistics, 2019, 13: 1-36.
[^5]: Halko N, Martinsson P G, Tropp J A. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions[J]. SIAM review, 2011, 53(2): 217-288.
