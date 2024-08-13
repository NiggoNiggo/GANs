import torchvision.transforms as transforms
from torch import nn, optim

import json
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname("CGAN")))

from CGAN.cgan_discriminator import Discriminator
from CGAN.cgan_generator import Generator
from Base_Models.conditional_data_loader import CustomDataset








cgan_dict = {
    "Generator":"Generator",
    "Generator_params":{
        "num_classes":7,
        "num_layers":5,
        "in_channels":[100,1024,512,256,128],
        "out_channels":[512,256,128,64,3],
        "kernel_sizes":[4,4,4,4,4],
        "strides":[1,2,2,2,2],
        "paddings":[0,1,1,1,1],
        "batchnorm":False
    },
    "Discriminator":"Discriminator",
    "Discriminator_params":{
        "num_classes":7,
        "num_layers":5,
        "in_channels":[3,64,128,256,512],
        "out_channels":[64,128,256,512,1],
        "kernel_sizes":[4,4,4,4,4],
        "strides":[2,2,2,2,1],
        "paddings":[1,1,1,1,0]
    },
    "Dataset":"CustomDataset",
    "loss_fn":"nn.BCELoss",
    "gen_optimizer":"optim.Adam",
    "disc_optimizer":"optim.Adam",
    "epochs":2,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "betas_dcgan":(0.5,0.999),
    "data_path":r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Images\Yellyfish",
    "batch_size":128,
    "latent_space":100,
    "lr_dcgan":2e-5,
    "lr_wgan":1e-4,
    "img_size":128,
    "transforms":"transforms",
    "lam":10,
    "n_crit":5,
    "alpha":0.0001,
    "betas_wgan":(0,0.9)
}

def load_parameters(filepath):
    with open(filepath, 'r') as f:
        params = json.load(f)
    
    # Resolve the class/function references
    params["Generator"] = cgan_mapping[params["Generator"]]
    params["Discriminator"] = cgan_mapping[params["Discriminator"]]
    params["Dataset"] = cgan_mapping[params["Dataset"]]
    params["loss_fn"] = cgan_mapping[params["loss_fn"]]()
    params["gen_optimizer"] = cgan_mapping[params["gen_optimizer"]]
    params["disc_optimizer"] = cgan_mapping[params["disc_optimizer"]]
    params["transforms"] = cgan_mapping[params["transforms"]]
    params["Critiker"] = cgan_mapping[params["Critiker"]]
    
    return params

cgan_mapping = {
    "Generator": Generator,
    "Discriminator": Discriminator,
    "CustomDataset": CustomDataset,
    "nn.BCELoss": nn.BCELoss,
    "optim.Adam": optim.Adam,
    "transforms":transforms.Compose([transforms.Resize(cgan_dict["img_size"]),
                                     transforms.ToTensor(),
                                     transforms.CenterCrop(cgan_dict["img_size"]),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

if __name__ == "__main__":
    with open('cgan_parameters.json', 'w') as f:
        json.dump(cgan_dict, f)