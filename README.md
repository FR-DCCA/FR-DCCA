# FR-DCCA

> This is code for paper  ‘Addressing Contradiction between Reconstruction and Correlation Maximization in Deep Canonical Correlation Autoencoders’
>
> We have published a reproducible demo and one dataset here.
>

## 1. How to run

>First, run 'mnist/wrtie_data_mnist.py' to write the data for 'train\val\test' into 'dataset/mnist' 
>
>Then, run 'mnist/full_recon_classify.py' on Noisy Mnist dataset for classificaation;
>
>run 'mnist/full_recon_cluster.py' on Noisy Mnist dataset for clustering.
>

## 2. Data Availablity

> Noisy Mnist data: original data can be download from 'https://ttic.uchicago.edu/~wwang5/dccae.html'， 
> and then you have to put it into 'dataset/'.

## 3. Requirements

>scikit-learn==0.20.0
>
>pytorch==1.0.0 
>
>torchvision==0.4.0
>
>numpy==1.19.5
