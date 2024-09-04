## This file contains various implementations of GANs.
As well there are various base classes implementet for training and validation the gans. This project is made in pytorch.

## Overview
This Project contains some classic GAN structures and helper functions to train dem. With this project is it possible to train DCGAN, CGAN, SpecGAN, WaveGAN, WGAN GP and more is coming later. These GANS has successfully implementet and generated audio and images. There are different ways to run this Project. You can import every gan on it's own and make run the training routine. Or you just use the *main.py* file and set up your varaiables.




## Install Dependencies and Requirements
Follow along the *requirements.txt* file and maybe install a suitable torch and cuda version for your operatin system.

## Structure of Files
The folder Base_Models contains various helper function and base classes. Here you can find the dataloaders and the custom layers for building your own models. The dataloaders are sperated in audio or image data.

Every GAN is structuerd as follows. One Folder with GAN name in Big letters contains a Generator and Discriminator python file. Additionally ther is one train file called similar to the folder. This file is a subclass of *gan_base.py* that contains the training routine in genral and some prediction and validation methodes. The train_one_epoch() Methode is not implementet and should be implementet for every GAN specificlly. But in WGAN GP and DCGAN the training routines for Wasserstein GAN with gradient penalty and the normal GAN training routine are implementet. If you want to use them you can inheriate your new GAN from one of these classes. Mostly the GAN folde contain additionally files to implement a conditonal version of the given GAN. This is not done for every GAN so far.

The Utils folder contains some helper functions that you need in order to get full success in training gans but are not relatet to the rest of this implementations.


## Available GANs so far
- DCGAN 
- WGAN GP
- CGAN~(not really)
- SpecGAN
- WaveGAN

## WaveGAN 
Wavegan is implementet with conditional training, that the labels can be given as well to the network. As well you can change the length of the wavegan to 1 or 4s.


# Usage

## Change Parameters
To change parameters you have to change the parameters in *Utils\parameters.py* or you can run the main file with the necessary arguments from the command line. If you want to load just one model and use it in your own script you have to import the file with the name of the gan similar to the folder name. Then the arguments are loaded from default values from the *Utils\parameters.py* file.

## Installation
You can pip install this repository, but maybe you have to load pytorch and cuda in your needed version. Because a virtual environment can cause error and not find a suitable cuda version and you have to train the models on cpu~(very slow).


## Notes
It is not completely working unless you change your paths in some files. This is going to be my nect task to general it to use it with everyones system. Additoinally there are some new GANs coming soon to this project

## Citations


- **DCGAN**: Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv preprint arXiv:1511.06434. [Paper](https://arxiv.org/abs/1511.06434)

- **CGAN**: Mirza, M., & Osindero, S. (2014). *Conditional Generative Adversarial Nets*. arXiv preprint arXiv:1411.1784. [Paper](https://arxiv.org/abs/1411.1784)

- **WGAN GP**: Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). *Improved Training of Wasserstein GANs*. In Advances in Neural Information Processing Systems (pp. 5767-5777). [Paper](https://arxiv.org/abs/1704.00028)

- **WaveGAN**: Donahue, C., McAuley, J., & Puckette, M. (2018). *Adversarial Audio Synthesis*. In ICLR. [Paper](https://arxiv.org/abs/1802.04208)

- **SpecGAN**: Donahue, C., McAuley, J., & Puckette, M. (2018). *Adversarial Audio Synthesis*. In ICLR. [Paper](https://arxiv.org/abs/1802.04208)
