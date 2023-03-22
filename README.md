# COGS 181 Final Project: Analysis of Different Generator Architectures for Generative Adversial Networks (GANs)

## Abstract

This paper describes the experiments of creating Generator models in Generative Adversial Networks (GANs) using different architectures and hyperparameters in order to find a model that can generate the most realistic images from random noise. Different methods such as Deep Convolutional layers, Conditional inputs, and Residual blocks are experimented, and the results are compared by looking at generator/discriminator loss and the quality of the generated images. The results show that a simple generator model with deep convolutional layers outperforms other models with conditional inputs and residual layers.  Although the use of convolutional layers and the increase in parameters can help improve image quality and generation performance, increasing the complexity (as seen with residual layers) does not necessarily lead to better results. The analysis of different generator architectures in GANs can help the field of image generation understand the tradeoff between model complexity and performance.

## About this repository

- `project.ipynb` is where all the testing was done.
- `compare.ipynb` is where graphs and image comparisons were done
- `pytorch_models/` is the folder of all the generator model architectures I have used for the experiments. I also created a `GANFactory` class that creates a discriminator and generator model based on the parameters you input. 
- `fixed_latent_inputs.pt` is a fixed 64-batch of latent vectors (128x1x1) that were used to track the progress of the generated images for each architecture. This was randomly generated in the beginning, and used for every experiment.
- `fixed_latent_labels.pt` is a fixed 64-batch of class labels ranging from 1-10, and used for GAN architectures that needed conditional inputs.