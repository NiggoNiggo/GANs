import json
import sys
import os 
import torch
from torch import optim
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.dirname("SpecGAN")))
from SpecGAN.spec_discriminator import SpecDiscriminator
from SpecGAN.spec_generator import SpecGenerator
from Base_Models.audio_data_loader import AudioDataset
from Base_Models.audio_transformer import SpecGANTransformer
from SpecGAN.conditional_spec_loader import MnistAudio


specgan_dict = {
    "Generator":"Generator",
    "Generator_params":{
        "num_layers":5,
        "c":1,
        "d":64
    },
    "Discriminator":"Discriminator",
    "Discriminator_params":{
        "num_layers":5,
        "c":1,
        "d":64
    },
    "transform":"SpecGANTransformer",
    "transform_params":{
        "n_fft":256,
        "win_length":256,
        "hop:length":128,
        "target_length":128,#16384
        "target_freq_bins":128,
        "target_fs":16000
    },
    "sr":16000,
    "Dataset":"Dataset",
    "Dataset_conditional_params":{
        "path":r"F:\DataSets\Audio\MINST",
        "num_classes":10,
        "transform":"SpecGANTransformer",
        "conditional":"True"
    },
    "Dataset_params":{
        "path":r"F:\DataSets\Audio\MINST",
        "num_classes":0,
        "transform":"SpecGANTransformer",
        "conditional":"False"
    },
    "gen_optimizer":"optim.Adam",
    "disc_optimizer":"optim.Adam",
    "epochs":50,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "betas":(0.5,0.9),
    "data_path":r"F:\DataSets\Audio\MINST",
    "batch_size":64,
    "latent_space":100,
    "lr":1e-5,
    "img_size":128,
    "lam":10,
    "n_crit":2,
    "alpha":0.0001,
    "dtype":"audio"
}

def load_parameters(filepath):
    with open(filepath, 'r') as f:
        params = json.load(f)
    
    # Resolve the class/function references
    params["Generator"] = specgan_mapping[params["Generator"]]
    params["Discriminator"] = specgan_mapping[params["Discriminator"]]
    params["Dataset"] = specgan_mapping[params["Dataset"]]
    params["gen_optimizer"] = specgan_mapping[params["gen_optimizer"]]
    params["disc_optimizer"] = specgan_mapping[params["disc_optimizer"]]
    params["Dataset_params"]["transform"] = specgan_mapping[params["transform"]](*specgan_dict["transform_params"].values())
    
    return params

specgan_mapping = {
    "Generator": SpecGenerator,
    "Discriminator": SpecDiscriminator,
    "AudioDataset": AudioDataset,
    "optim.Adam": optim.Adam,
    "Dataset":MnistAudio,
    "SpecGANTransformer":SpecGANTransformer
}


if __name__ == "__main__":
    with open('specgan_parameters.json', 'w') as f:
        json.dump(specgan_dict, f)
        
# utils paar bearbeitungsfunktionen machen und dann hier eingeben