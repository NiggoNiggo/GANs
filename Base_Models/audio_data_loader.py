from torch.utils.data import Dataset
from Base_Models.audio_transformer import SpecGANTransformer
import librosa
import torchaudio.transforms as T
import torch.nn.functional as F
import torchaudio
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

class AudioDataset(Dataset):
    def __init__(self,
                 path:str):
        super().__init__()
        self.path = path
        self.all_files = []
        self.target_length = 16384
        self.target_fs = 16000
        #iterate through every file
        for r,d,f in os.walk(self.path):
            #access all files
            all_files = [os.path.join(r,file) for file in f if file.endswith(".wav")]
            self.all_files.extend(all_files)
        
        self.transform = SpecGANTransformer(256,256,128,128,128)
        
        
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        # Load audio file using torchaudio
        data, fs = torchaudio.load(self.all_files[idx])
        
        # Ensure the audio is mono (single channel)
        if data.size(0) > 1:
            data = torch.mean(data, dim=0, keepdim=True)
        
        # Calculate the number of missing samples
        current_length = data.size(1)
        if current_length > self.target_length:
            # Truncate the audio
            data = data[:, :self.target_length]
        else:
            # Pad the audio with zeros
            missing = self.target_length - current_length
            padding = (0, missing)  # (left_pad, right_pad)
            data = torch.nn.functional.pad(data, padding)
        
        x = self.transform(data)
        return x

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



    
        

