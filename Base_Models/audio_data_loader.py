from torch.utils.data import Dataset
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_files = []
        self.target_length = 16384
        self.target_fs = 16000
        #iterate through every file
        for r,d,f in os.walk(self.path):
            #access all files
            all_files = [os.path.join(r,file) for file in f if file.endswith(".wav")]
            self.all_files.extend(all_files)
        
        self.transform = AudioTransform(256,256,128,128,128)
        
        
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
        
        x = self.transform(data).to(self.device)
        return x

    def num_classes(self):
        return 

    
        



class AudioTransform:
    def __init__(self, n_fft, win_length, hop_length, target_length, target_freq_bins):
        self.stft = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None)
        self.target_length = target_length
        self.target_freq_bins = target_freq_bins
        self.target_fs = 16000

    def __call__(self, data):
        data = data.float()
        # Compute spectrogram
        spec = torch.abs(self.stft(data))
        
        # Ensure the spectrogram has the correct number of frequency bins
        if spec.size(1) > self.target_freq_bins:
            # Truncate the frequency bins
            spec = spec[:, :self.target_freq_bins, :]
        elif spec.size(1) < self.target_freq_bins:
            # Pad the frequency bins
            padding = (0, 0, 0, self.target_freq_bins - spec.size(1))  # (left, right, top, bottom)
            spec = F.pad(spec, padding)
        
        # Ensure the spectrogram has the correct number of time steps
        if spec.size(2) > self.target_length:
            # Truncate the time steps
            spec = spec[:, :, :self.target_length]
        elif spec.size(2) < self.target_length:
            # Pad the time steps
            padding = (0, self.target_length - spec.size(2))  # (left, right)
            spec = F.pad(spec, padding)
        
        return spec
    

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     path = r"C:\Users\analf\Desktop\Studium\Learn_NN\Datasets\Cats_Dogs_Audio\cats_dogs"
#     dataset = AudioDataset(path)
    
#     dataset[1]
