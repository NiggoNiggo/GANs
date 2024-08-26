import os
import librosa
import torchaudio.transforms as T
import torchaudio
import torch


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MnistAudio(Dataset):
    def __init__(self,
                 path:str,
                 num_classes,
                 transforms,
                 conditional:bool=False):
        super().__init__()
        self.path = path
        self.num_classes = num_classes
        self.transforms = transforms
        self.conditional = conditional
        if self.conditional == True:
            self.data = {x:[] for x in range(num_classes)}#contains label and data disct{str:list}
        self.load_data()
        
        
        
            
    def load_data(self):
        self.all_files = []
        for r,d,f in os.walk(self.path):
            self.all_files.extend([os.path.join(r,file) for file in f if file.endswith(".wav")])
        if self.conditional == True: 
            for file in self.all_files:
                #get first letter of the filename because it is the label
                label = int(file.split("\\")[-1][0])
                self.data[label].append(file)
                    
        
            
    
    def __len__(self):
        if self.conditional == True:
            return len(min([values for values in self.data.values()]))
        else:
            return len(self.all_files)
    
    def __getitem__(self,idx):
        if self.conditional == True:
            all_data = [values for values in self.data.values()]
            all_data = [item for sublist in all_data for item in sublist]
        else:
            all_data = self.all_files
        current_file = all_data[idx]
        data, fs = librosa.load(path=current_file,sr=16000,mono=True)

        data = torch.from_numpy(data).unsqueeze(0)
        data = self.transforms(data, fs)
        if self.conditional == True:
            current_label = int(current_file.split("\\")[-1][0])
            label = torch.tensor(current_label)
            return data, label
        else:
            return data


    def plot_spectrogram(self, spectrogram):
        """ Utility function to plot the spectrogram. """
        spectrogram = spectrogram.squeeze().numpy()  # Convert to numpy array
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect='auto', origin='lower')
        plt.title("Spectrogram")
        plt.colorbar()
        plt.show()
    def get_num_classes(self):
        return self.num_classes

