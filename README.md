# Analysis and Approximate Inference of Large Random Kronecker Graph
This repository contains code to reproduces the numerical results in the paper "*Analysis and Approximate Inference of Large Random Kronecker Graph*".


## About the code
* The repository `/code_Analysis_and_Approximate_Inference` contains 
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
  * sub-folder `/func` that contains 
    * **generate_PK.m** : generate probability matrix using K times Kronecker power
    * **generate_Theta.m**: generate coefficient matrix
    * **shuffle.m**: using a permutation matrix to shuffle a graph
    * **de_noise.m**: de-noise by constructing an estimator S_approx_shrink of SK
    * **de_noise_rsvd.m**: de-noise by constructing an estimator S_approx_shrink of SK using rsvd
    * **get_block.m**: random sampling for RNLA 
    * **solve_convex_relaxation_func.m**: the implement of Algorithm 3
  * sub-folder `/realgraph_datasets` that contains several chemical and social graph datasets[^9]: NCI1[^7], REDDIT-BINARY[^8], IMDB-BINARY[^8], PROTEINS_full[^6].
     
* The repository `/classification_task` contains 
  * **main.py**: test for classification_task, following the experiments in [^1]
  * **classes.py**: contains several modules for performing classification task in **main.py**
    
  
## Dependencies
The code in `/classification_task` relies on the following packages:
* [Python](https://www.python.org/): tested with version 3.7.10
* [Pytorch](https://pytorch.org/): tested with version 1.8.1
* [Pandas](https://pandas.pydata.org/): tested with version 1.3.5
* [Scikit-learn](https://scikit-learn.org/stable/): tested with version 1.0.2


## References
[^1]: Errica F, Podda M, Bacciu D, et al. A Fair Comparison of Graph Neural Networks for Graph Classification[C]//International Conference on Learning Representations. 2019.
[^2]: Blumensath T, Davies M E. Iterative thresholding for sparse approximations[J]. Journal of Fourier analysis and Applications, 2008, 14: 629-654.
[^3]: Jain P, Kar P. Non-convex optimization for machine learning[J]. Foundations and Trends® in Machine Learning, 2017, 10(3-4): 142-363.
[^4]: Slawski M, Ben-David E. Linear regression with sparsely permuted data[J]. Electronic Journal of Statistics, 2019, 13: 1-36.
[^5]: Halko N, Martinsson P G, Tropp J A. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions[J]. SIAM review, 2011, 53(2): 217-288.
[^6]: Borgwardt K M, Ong C S, Schönauer S, et al. Protein function prediction via graph kernels[J]. Bioinformatics, 2005, 21(suppl_1): i47-i56.
[^7]: Wale N, Watson I A, Karypis G. Comparison of descriptor spaces for chemical compound retrieval and classification[J]. Knowledge and Information Systems, 2008, 14: 347-375.
[^8]: Yanardag P, Vishwanathan S V N. Deep graph kernels[C]//Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining. 2015: 1365-1374.
[^9]: Kersting, K., Kriege, N. M., Morris, C., Mutzel, P., and Neumann, M. Benchmark data sets for graph kernels, 2016. URL http://graphkernels.cs.tu-dortmund.de.

