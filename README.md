
# GAN-PyTorch


This project aims to reproduce some results of the paper : [Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

The goal is to experiment and get results on the factors that influence the **stability** and the **performance** of **Generative adversarial networks (GANs)**. The implementation is in **PyTorch** using torch.nn, torch.optim and torchvision libraires mostly. I chose to start with the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit database.

 

# Table of contents

 1. **Introduction to Generative Adversarial Networks**
	 1. Generative Models
	 2. GAN 
 2. **Experimentation on MNIST database**
	 1. Implementation details
	 2. Stability issues
 3. **Results**
 4. **Next things to do**


# Introduction to Generative Adversarial Networks

## Generative models

The most know machine learning algorithm are usually **discriminative models**, for example classifiers. Classifiers aims to predict the class of an example given as input to the algorithm. 

More mathematically, discriminative models approximate the value of P(Y|X=x), whereas **generative models approximate P(X,Y) the joint distribution**.

For classifiers, P(Y|X=x) is the probability of being in particular class (output of the classifier) knowing that X=x which means that the example is x (input of the classifier).

Basically, generative models try to **approximate the "true" distribution of the data**. Thanks to this approximation, we are able to artificially generate new examples.

*Remark: Classifiers and generative models are not opposites. It is possible to build a classifier with a generative model because : P(Y=y|X=x) = P(Y=y,X=x) / P(X=x).*

## GAN


Basically, **GAN** is one of the generative models. It offers simple and powerful framework to fit the probability distribution we talked.

### How does it work ?

The idea is to build **two neural networks** : one that generates examples and one that discriminates false example (coming from the generator) and true examples (coming from the data) and then, to make them **compete against each other** until the generator is "good" enough, which means that it approximates the data distribution well. If so, the discriminating network should be fooled by the generated examples.

The generative algorithm takes noise **z** (following **normal distribution p_normal**) as input and outputs **G(z)**.

The discriminative algorithm takes examples x from the dataset and G(z) the generated examples and outputs D(x) or D(G(z)) the probability of coming from the dataset.

### Equation of the problem

Basically, the **discriminative network** training aims to **maximize D(x)**, or log(D(x)) and **minimize D(G(z))**, or maximize log(1-D(G(z)).

Whereas, the **generative network** training aims to **maximize D(G(z))**, or minimize log(1-D(G(z)).


Mathematically, it is a minimax game : **min_G max_D E(D(X)) + E(log(1-D(G(Z))))**

With: 
 - **X** follow **p_data** data distribution (*"true distribution"*)
 - **Z** follow **p_g** generated distribution
 - **G** generative model
 - **D** discriminative model

# Experimentation on MNIST database

## Implementation details

### Dataset
 - Zeros of the MNIST dataset
 - 5923 examples
 - 28*28 = 784 pixels images
 - **Normalized** (-1 <-> 1)

### Generative model
 - **3-layer perceptron** (64 - 128 - 512 - 784)
 - Activation functions
	 - **Leaky ReLU** on hidden layers
	 - **Tanh** on output layer
 - **Binary cross entropy loss**
 
### Discriminative model
 - **2-layer perceptron** (784 - 256 - 1)
 - Activation functions
	 - **Leaky ReLU** on hidden layer
	 - **Sigmoid** on output layer
 - **Binary cross entropy loss**

### Hyperparameters
 - Learning rate = 0.001 (generative network) / 0.05 (discriminative network)
 - Mini-batch size = 16 (generative network) / 32 (discriminative network)
 - Number of iterations = 250
 - No. of D training/No. of G training ratio = 1:1

### How does the algorithm works ?

The algorithm consist in a **loop of these two successive trainings** :

 - One iteration of **discriminative network training** consists in:
	 - Generating 16 examples with G and normal noise.
	 - Taking 16 examples from the data.
	 - Doing one step of **backpropagation** with a batch composed of these 32 examples, with **0 as labels for the generated example and 1 for the examples coming from the data.**
	 
- On the contrary, one iteration of **generative network training** consists in:
	 - Generating 16 examples with G and normal noise.
	 - Calculating **D(G(z))** for these examples and the loss with **1 as labels for all the examples** (the aim is to train G to fool D).
	 - Backpropagating the errors, but only updating the parameters of G.


## Stability issues
*Work in progress*

# Results
*Work in progress*

# Next things to do

 - DCGAN (using CNN as discriminator and a deconvolutional network as generator)
 - Experimenting on CIFAR-10 database
 
## Built with
-   Python
-   PyTorch