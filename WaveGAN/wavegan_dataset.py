from torch.utils.data import Dataset
import os
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
import soundfile as sf



class WaveDataset(Dataset):
    def __init__(self,
                 path,
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
    def __init__(self,
                 path,
                 transform):
        super().__init__()
        self.path = path#to a csv file that contains path and label
        self.transform = transform
        self.data = pd.read_csv(self.path,index_col=False)

    def __len__(self):
        return len(self.data)

    def _return_num_classes(self):
        return len(pd.unique(self.data))

    def __getitem__(self,idx):
        data, fs = torchaudio.load(self.data.Filename[idx])
        label = self.data.Label[idx]
        data = self.transform(data)
        return data, label

        