# Inference_of_Kronecker_Graph
## About the code

* `func` contains 
  * `code/compression/cifar10` : generate probability matrix using K times Kronecker power
    * input
      * -P1 Kronecker initiator
      * -K I
    * output:
  * **generate_Theta.m**: generate linearized probability matrix 
    * input: 
    * output:


* `code/spectral_characteristics` **(Experiment 1)** contains
  * **tilde_CK.py** for verifying the consistency of spectrum distribution for **theoretical** (calculated with our theorem results) and **practical** (calculated by the original definition) conjugate kernel(CK).
  * **plot_eigen.py** for ploting eigenvalues and eigenvectors for a given matrix.

* `code/equation_solve` contains
  * **solve_equation.py** for solving equatiosn to define parameters of activation functions

* `code/expect_cal` contains
  * **expect_calculate.py** for expect calculated by numerical integration
  * **expect_calculate_math.py** for expect calculated with analytical expression
 
* `code/model_define` contains
  * **model.py** for model defining
 
* `code/utils` contains
  * **activation_numpy.py** for activations defined with numpy
  * **activation_tensor.py** for activations defined with torch
  * **data_prepare.py** for data preparation, containing data sampled from MNIST/CIFAR10 and generated GMM data
  * **utils.py** for some more utils 

