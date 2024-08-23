## This file contains various implementations of GANs.
As well there are various base classes implementet for training and validation the gans. This project is made in pytorch.

## Overview
This Project contains some classic GAN structures and helper functions to train dem. With this project is it possible to train DCGAN, CGAN, SpecGAN, WaveGAN, WGAN GP and more is coming later. These GANS has successfully implementet and generated audio and images.

## Requirements
see the requirements.txt file for further informations

## Installation

## Install Dependencies

## Structure of Files
The folder Base_Models contains various helper function and base classes. Here you can find the dataloaders and the custom layers for building your own models. The dataloaders are sperated in audio or image data.

Every GAN is structuerd as follows. One Folder with GAN name in Big letters contains a Generator and Discriminator python file. Additionally ther is one train file called similar to the folder. This file is a subclass of *gan_base.py* that contains the training routine in genral and some prediction and validation methodes. The train_one_epoch() Methode is not implementet and should be implementet for every GAN specificlly. But in WGAN GP and DCGAN the training routines for Wasserstein GAN with gradient penalty and the normal GAN training routine are implementet. If you want to use them you can inheriate your new GAN from one of these classes. Mostly the GAN folde contain additionally files to implement a conditonal version of the given GAN. This is not done for every GAN so far.

The Utils folder contains some helper functions that you need in order to get full success in training gans but are not relatet to the rest of this implementations.


## Descriptions of GANs
- DCGAN 
- WGAN GP
- CGAN
- SpecGAN
- WaveGAN


# Usage

## Change Parameters
#this is going to be replaced by argparser soon


## Citations

Wenn du diese Implementationen in deiner Forschung oder Projekten verwendest, zitiere bitte die entsprechenden Arbeiten:

- **DCGAN**: Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv preprint arXiv:1511.06434. [Paper](https://arxiv.org/abs/1511.06434)

- **CGAN**: Mirza, M., & Osindero, S. (2014). *Conditional Generative Adversarial Nets*. arXiv preprint arXiv:1411.1784. [Paper](https://arxiv.org/abs/1411.1784)

- **WGAN GP**: Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). *Improved Training of Wasserstein GANs*. In Advances in Neural Information Processing Systems (pp. 5767-5777). [Paper](https://arxiv.org/abs/1704.00028)

- **WaveGAN**: Donahue, C., McAuley, J., & Puckette, M. (2018). *Adversarial Audio Synthesis*. In ICLR. [Paper](https://arxiv.org/abs/1802.04208)

- **SpecGAN**: Donahue, C., McAuley, J., & Puckette, M. (2018). *Adversarial Audio Synthesis*. In ICLR. [Paper](https://arxiv.org/abs/1802.04208)
