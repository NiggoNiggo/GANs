from torch.utils.data import Dataset
import os
import torchaudio
import librosa

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