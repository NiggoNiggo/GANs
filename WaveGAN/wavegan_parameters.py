import json
import sys
import os 
import torch
from torch import optim
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.dirname("WaveGAN")))
from WaveGAN.wavegan_dataset import WaveDataset
from Base_Models.audio_transformer import WaveNormalizer

from WaveGAN.wave_discriminator import WaveDiscriminator
from WaveGAN.wave_generator import WaveGenerator




wavegan_dict = {
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
    "sr":16000,
    "Dataset":"WaveDataset",
    "Dataset_params":{
        "path":r"C:\Users\analf\Desktop\Datasets_And_Results\Datasets\RS6",
        "transform":"waveganTransformer"
    },
    "gen_optimizer":"optim.Adam",
    "disc_optimizer":"optim.Adam",
    "epochs":1000,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "betas":(0.5,0.9),
    "data_path":r"C:\Users\analf\Desktop\Datasets_And_Results\Datasets\RS6",
    "batch_size":64,
    "latent_space":100,
    "lr":1e-4,
    "lam":10,
    "n_crit":5,
    "alpha":0.0001,
    "dtype":"audio"
}

def load_parameters(filepath):
    with open(filepath, 'r') as f:
        params = json.load(f)
    
    # Resolve the class/function references
    params["Generator"] = wavegan_mapping[params["Generator"]]
    params["Discriminator"] = wavegan_mapping[params["Discriminator"]]
    params["Dataset"] = wavegan_mapping[params["Dataset"]]
    params["gen_optimizer"] = wavegan_mapping[params["gen_optimizer"]]
    params["disc_optimizer"] = wavegan_mapping[params["disc_optimizer"]]
    params["Dataset_params"]["transform"] = wavegan_mapping[params["Dataset_params"]["transform"]]()
    
    return params

wavegan_mapping = {
    "Generator": WaveGenerator,
    "Discriminator": WaveDiscriminator,
    "optim.Adam": optim.Adam,
    "WaveDataset":WaveDataset,
    "waveganTransformer":WaveNormalizer
}


if __name__ == "__main__":
    with open('wavegan_parameters.json', 'w') as f:
        json.dump(wavegan_dict, f)
        
# utils paar bearbeitungsfunktionen machen und dann hier eingeben