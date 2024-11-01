from torch.utils.data import Dataset
import os
import torchaudio
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import soundfile as sf



class WaveDataset(Dataset):
    """Dataset for WaveGAN.

        Parameters
        ----------
        path : str
            path to the folder with the training material
        transform : torchvision.transforms
            transformation to do on the audio files
        """
    def __init__(self,
                 path:str,
                 transform):
        super().__init__()
        self.path = path
        self.transform = transform
        self.all_data = []
        for r,d,f in os.walk(path):
            self.all_data.extend([os.path.join(r,file) for file in f if file.endswith(".wav")])

    def __len__(self):
        return len(self.all_data)


    
    def __getitem__(self, idx):
        data, fs = torchaudio.load(self.all_data[idx])
        data = data[0]
        data = self.transform(data)
        return data

class ConditionalWaveDataset(Dataset):
    """Dataset for conditional WaveGAN.

    Parameters
    ----------
    path : str
        path to the folder with the training material
    transform : torchvision.transforms
        transformation to do on the audio files
    """
    def __init__(self,
                 path,
                 transform):
        super().__init__()
        self.path = path#to a csv file that contains path and label
        self.transform = transform
        self.data = pd.read_csv(os.path.normpath(self.path))[:None]

    def __len__(self):
        return len(self.data)

    def _return_num_classes(self):
        return len(pd.unique(self.data))

    def __getitem__(self,idx):
        data, fs = torchaudio.load(os.path.normpath(self.data.Filename[idx]))
        label = self.data.Label[idx]
        data = self.transform(data)
        return data, label

        