import torchvision.transforms as transforms
from torch import nn, optim

import json
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname("DCGAN")))

from DCGAN.dcgan_discriminator import Discriminator
from DCGAN.dcgan_generator import Generator
from WGAN_GP.wgan_gp_critic import Critiker
from Base_Models.image_data_loader import CustomDataset








dcgan_dict = {
    "Generator":"Generator",
    "Generator_params":{
        "num_layers":6,
        "in_channels":[100,1024,512,256,128,64],
        "out_channels":[1024,512,256,128,64,3],
        "kernel_sizes":[4,4,4,4,4,4],
        "strides":[1,2,2,2,2,2],
        "paddings":[0,1,1,1,1,1],
        "batchnorm":False
    },
    "Critiker":"Critiker",
    "Discriminator":"Discriminator",
    "Discriminator_params":{
        "num_layers":6,
        "in_channels":[3,64,128,256,512,1024],
        "out_channels":[64,128,256,512,1024,1],
        "kernel_sizes":[4,4,4,4,4,4],
        "strides":[2,2,2,2,2,1],
        "paddings":[1,1,1,1,1,0]
    },
    "Dataset":"CustomDataset",
    "loss_fn":"nn.BCELoss",
    "gen_optimizer":"optim.Adam",
    "disc_optimizer":"optim.Adam",
    "epochs":60,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "betas_dcgan":(0.5,0.999),
    "data_path":r"F:\DataSets\Images\Zebra",
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
    params["Generator"] = dcgan_mapping[params["Generator"]]
    params["Discriminator"] = dcgan_mapping[params["Discriminator"]]
    params["Dataset"] = dcgan_mapping[params["Dataset"]]
    params["loss_fn"] = dcgan_mapping[params["loss_fn"]]()
    params["gen_optimizer"] = dcgan_mapping[params["gen_optimizer"]]
    params["disc_optimizer"] = dcgan_mapping[params["disc_optimizer"]]
    params["transforms"] = dcgan_mapping[params["transforms"]]
    params["Critiker"] = dcgan_mapping[params["Critiker"]]
    
    return params

dcgan_mapping = {
    "Generator": Generator,
    "Discriminator": Discriminator,
    "Critiker":Critiker,
    "CustomDataset": CustomDataset,
    "nn.BCELoss": nn.BCELoss,
    "optim.Adam": optim.Adam,
    "transforms":transforms.Compose([transforms.Resize(dcgan_dict["img_size"]),
                                     transforms.ToTensor(),
                                     transforms.CenterCrop(dcgan_dict["img_size"]),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

if __name__ == "__main__":
    with open('dcgan_parameters.json', 'w') as f:
        json.dump(dcgan_dict, f)